#!/usr/bin/env python3
"""Build binding and glycan-structure artifacts from supplementary tables.

This script implements an end-to-end preprocessing pipeline:
1. (Optional) unzip supplementary archive
2. Load S1/S2/S3 tables (legacy .xls supported without xlrd)
3. Convert binding matrix to long-form triples
4. Clean targets, normalize per-virus, create classification labels
5. Build glycan structure table with provenance
6. Parse glycan strings into lightweight topology annotations
7. Precompute reproducible split columns

Outputs (default: ./artifacts):
- viral_glycan_bind_long.parquet
- glycans.parquet
- glycans_tokenized.parquet
- glycan_column_mapping.parquet
- preprocessing_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd


FREESECT = 0xFFFFFFFF
ENDOFCHAIN = 0xFFFFFFFE

BRANCH_LINKAGE_RE = re.compile(r"[ab][0-9?]-[0-9?]", re.IGNORECASE)
LINKAGE_SUFFIX_RE = re.compile(r"([ab][0-9?]-[0-9?]|[ab])$", re.IGNORECASE)
SPACER_SUFFIX_RE = re.compile(r"-sp\d+.*$", re.IGNORECASE)


@dataclass
class CfbDirEntry:
    name: str
    obj_type: int
    start_sector: int
    size: int


@dataclass
class ParseArtifacts:
    long_df: pd.DataFrame
    glycans_df: pd.DataFrame
    tokenized_df: pd.DataFrame
    mapping_df: pd.DataFrame
    report: dict[str, Any]


def stable_hash_fraction(key: str) -> float:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return (int(h[:12], 16) % 1_000_000) / 1_000_000.0


def make_unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        base = col.strip() if col and isinstance(col, str) else "unnamed"
        if not base:
            base = "unnamed"
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count}")
    return out


def read_workbook_stream_from_xls(path: Path) -> bytes:
    data = path.read_bytes()
    header = data[:512]
    if header[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        raise ValueError(f"Not a CFB file: {path}")

    sector_size = 1 << struct.unpack_from("<H", header, 0x1E)[0]
    mini_sector_size = 1 << struct.unpack_from("<H", header, 0x20)[0]
    num_fat_sectors = struct.unpack_from("<I", header, 0x2C)[0]
    first_dir_sector = struct.unpack_from("<I", header, 0x30)[0]
    mini_stream_cutoff = struct.unpack_from("<I", header, 0x38)[0]
    first_minifat_sector = struct.unpack_from("<I", header, 0x3C)[0]
    num_minifat_sectors = struct.unpack_from("<I", header, 0x40)[0]
    first_difat_sector = struct.unpack_from("<I", header, 0x44)[0]
    num_difat_sectors = struct.unpack_from("<I", header, 0x48)[0]
    difat = list(struct.unpack_from("<109I", header, 0x4C))

    def read_sector(sector_id: int) -> bytes:
        offset = (sector_id + 1) * sector_size
        return data[offset : offset + sector_size]

    difat_entries = [d for d in difat if d != FREESECT]
    next_difat = first_difat_sector
    for _ in range(num_difat_sectors):
        sec = read_sector(next_difat)
        vals = struct.unpack(f"<{(sector_size // 4) - 1}I", sec[:-4])
        difat_entries.extend(v for v in vals if v != FREESECT)
        next_difat = struct.unpack("<I", sec[-4:])[0]
        if next_difat == ENDOFCHAIN:
            break

    fat: list[int] = []
    for sid in difat_entries[:num_fat_sectors]:
        fat.extend(struct.unpack(f"<{sector_size // 4}I", read_sector(sid)))

    def chain(start_sector: int) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        sid = start_sector
        while sid not in (ENDOFCHAIN, FREESECT):
            if sid in seen:
                break
            seen.add(sid)
            out.append(sid)
            if sid >= len(fat):
                break
            sid = fat[sid]
        return out

    dir_stream = b"".join(read_sector(s) for s in chain(first_dir_sector))
    entries: list[CfbDirEntry] = []
    for i in range(0, len(dir_stream), 128):
        rec = dir_stream[i : i + 128]
        if len(rec) < 128:
            break
        name_len = struct.unpack_from("<H", rec, 64)[0]
        if name_len >= 2:
            name = rec[: max(0, name_len - 2)].decode("utf-16le", errors="ignore")
        else:
            name = ""
        obj_type = rec[66]
        start_sector = struct.unpack_from("<I", rec, 116)[0]
        size = struct.unpack_from("<Q", rec, 120)[0]
        entries.append(CfbDirEntry(name, obj_type, start_sector, size))

    root = entries[0]
    root_stream = b"".join(read_sector(s) for s in chain(root.start_sector))[: root.size]

    minifat: list[int] = []
    if num_minifat_sectors and first_minifat_sector != ENDOFCHAIN:
        for sid in chain(first_minifat_sector):
            minifat.extend(struct.unpack(f"<{sector_size // 4}I", read_sector(sid)))

    def read_stream(entry: CfbDirEntry) -> bytes:
        if entry.size < mini_stream_cutoff and entry.obj_type == 2:
            out: list[bytes] = []
            sid = entry.start_sector
            seen: set[int] = set()
            while sid not in (ENDOFCHAIN, FREESECT):
                if sid in seen:
                    break
                seen.add(sid)
                off = sid * mini_sector_size
                out.append(root_stream[off : off + mini_sector_size])
                sid = minifat[sid] if sid < len(minifat) else ENDOFCHAIN
            return b"".join(out)[: entry.size]
        return b"".join(read_sector(s) for s in chain(entry.start_sector))[: entry.size]

    wb = next((e for e in entries if e.obj_type == 2 and e.name in {"Workbook", "Book"}), None)
    if wb is None:
        raise ValueError(f"Workbook stream not found in {path}")
    return read_stream(wb)


def iter_biff_records(stream: bytes) -> Iterator[tuple[int, int, bytes]]:
    i = 0
    n = len(stream)
    while i + 4 <= n:
        rec_id, rec_len = struct.unpack_from("<HH", stream, i)
        i += 4
        payload = stream[i : i + rec_len]
        if len(payload) < rec_len:
            break
        yield i - 4, rec_id, payload
        i += rec_len


class ContinueReader:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.chunk_i = 0
        self.pos = 0

    def _next_chunk(self) -> None:
        self.chunk_i += 1
        self.pos = 0

    def read(self, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            if self.chunk_i >= len(self.chunks):
                raise EOFError("Unexpected end of CONTINUE chunks")
            chunk = self.chunks[self.chunk_i]
            remaining = len(chunk) - self.pos
            if remaining <= 0:
                self._next_chunk()
                continue
            take = min(n, remaining)
            out.extend(chunk[self.pos : self.pos + take])
            self.pos += take
            n -= take
        return bytes(out)

    def read_u8(self) -> int:
        return self.read(1)[0]

    def read_u16(self) -> int:
        return struct.unpack("<H", self.read(2))[0]

    def read_u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def skip(self, n: int) -> None:
        if n > 0:
            self.read(n)

    def read_xl_chars(self, char_count: int, is_16bit: bool) -> str:
        parts: list[str] = []
        while char_count > 0:
            if self.chunk_i >= len(self.chunks):
                break
            chunk = self.chunks[self.chunk_i]
            remaining = len(chunk) - self.pos
            if remaining <= 0:
                self._next_chunk()
                if self.chunk_i >= len(self.chunks):
                    break
                is_16bit = bool(self.read_u8() & 0x01)
                continue
            bytes_per_char = 2 if is_16bit else 1
            max_chars = remaining // bytes_per_char
            if max_chars <= 0:
                self._next_chunk()
                if self.chunk_i >= len(self.chunks):
                    break
                is_16bit = bool(self.read_u8() & 0x01)
                continue
            take = min(char_count, max_chars)
            raw = self.read(take * bytes_per_char)
            parts.append(raw.decode("utf-16le" if is_16bit else "latin1", errors="replace"))
            char_count -= take
        return "".join(parts)


def parse_shared_strings(records: list[tuple[int, int, bytes]]) -> list[str]:
    sst_start = next((i for i, (_, rid, _) in enumerate(records) if rid == 0x00FC), None)
    if sst_start is None:
        return []
    chunks = [records[sst_start][2]]
    j = sst_start + 1
    while j < len(records) and records[j][1] == 0x003C:
        chunks.append(records[j][2])
        j += 1

    rd = ContinueReader(chunks)
    _total = rd.read_u32()
    unique = rd.read_u32()
    sst: list[str] = []
    for _ in range(unique):
        cch = rd.read_u16()
        flags = rd.read_u8()
        is_16bit = bool(flags & 0x01)
        has_ext = bool(flags & 0x04)
        has_rich = bool(flags & 0x08)
        c_run = rd.read_u16() if has_rich else 0
        cb_ext = rd.read_u32() if has_ext else 0
        text = rd.read_xl_chars(cch, is_16bit)
        if c_run:
            rd.skip(c_run * 4)
        if cb_ext:
            rd.skip(cb_ext)
        sst.append(text)
    return sst


def decode_rk_value(rk: int) -> float:
    is_integer = bool(rk & 0x01)
    divide_by_100 = bool(rk & 0x02)
    if is_integer:
        value = rk >> 2
        if value & (1 << 29):
            value -= 1 << 30
        out = float(value)
    else:
        raw = (rk & 0xFFFFFFFC) << 32
        out = struct.unpack("<d", struct.pack("<Q", raw))[0]
    return out / 100.0 if divide_by_100 else out


def parse_first_sheet_from_biff(workbook_stream: bytes) -> tuple[str, list[list[Any]]]:
    records = list(iter_biff_records(workbook_stream))
    sst = parse_shared_strings(records)

    boundsheets: list[tuple[str, int]] = []
    for _, rec_id, payload in records:
        if rec_id != 0x0085:
            continue
        bof_offset = struct.unpack_from("<I", payload, 0)[0]
        name_len = payload[6]
        grbit = payload[7]
        if grbit & 0x01:
            name = payload[8 : 8 + (name_len * 2)].decode("utf-16le", errors="replace")
        else:
            name = payload[8 : 8 + name_len].decode("latin1", errors="replace")
        boundsheets.append((name, bof_offset))

    if not boundsheets:
        raise ValueError("No sheet found in workbook")

    boundsheets.sort(key=lambda x: x[1])
    sheet_name, sheet_start = boundsheets[0]
    sheet_end = boundsheets[1][1] if len(boundsheets) > 1 else len(workbook_stream)

    cells: dict[tuple[int, int], Any] = {}
    nrows: int | None = None
    ncols: int | None = None
    max_row = -1
    max_col = -1

    i = sheet_start
    while i + 4 <= sheet_end:
        rec_id, rec_len = struct.unpack_from("<HH", workbook_stream, i)
        i += 4
        payload = workbook_stream[i : i + rec_len]
        i += rec_len
        if rec_id == 0x000A:
            break
        if rec_id == 0x0200 and len(payload) >= 12:
            # DIMENSIONS: rwMic(4), rwMac(4), colMic(2), colMac(2)
            nrows = struct.unpack_from("<I", payload, 4)[0]
            ncols = struct.unpack_from("<H", payload, 10)[0]
            continue
        if rec_id == 0x00FD and len(payload) >= 10:
            row, col = struct.unpack_from("<HH", payload, 0)
            sst_idx = struct.unpack_from("<I", payload, 6)[0]
            value = sst[sst_idx] if sst_idx < len(sst) else None
            cells[(row, col)] = value
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            continue
        if rec_id == 0x0203 and len(payload) >= 14:
            row, col = struct.unpack_from("<HH", payload, 0)
            value = struct.unpack_from("<d", payload, 6)[0]
            cells[(row, col)] = value
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            continue
        if rec_id == 0x027E and len(payload) >= 10:
            row, col = struct.unpack_from("<HH", payload, 0)
            rk = struct.unpack_from("<I", payload, 6)[0]
            cells[(row, col)] = decode_rk_value(rk)
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            continue
        if rec_id == 0x00BD and len(payload) >= 6:
            row = struct.unpack_from("<H", payload, 0)[0]
            col_first = struct.unpack_from("<H", payload, 2)[0]
            col_last = struct.unpack_from("<H", payload, len(payload) - 2)[0]
            p = 4
            for col in range(col_first, col_last + 1):
                rk = struct.unpack_from("<I", payload, p + 2)[0]
                cells[(row, col)] = decode_rk_value(rk)
                max_row = max(max_row, row)
                max_col = max(max_col, col)
                p += 6

    out_rows = nrows if nrows is not None else max_row + 1
    out_cols = ncols if ncols is not None else max_col + 1
    matrix = [[None for _ in range(out_cols)] for _ in range(out_rows)]
    for (row, col), value in cells.items():
        if row < out_rows and col < out_cols:
            matrix[row][col] = value
    return sheet_name, matrix


def load_first_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xls":
        _, matrix = parse_first_sheet_from_biff(read_workbook_stream_from_xls(path))
        return pd.DataFrame(matrix)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name=0, header=None)
    if suffix == ".csv":
        return pd.read_csv(path, header=None)
    raise ValueError(f"Unsupported input table format: {path}")


def detect_table_paths(
    input_dir: Path,
    explicit_s1: Path | None,
    explicit_s2: Path | None,
    explicit_s3: Path | None,
) -> tuple[Path, Path, Path]:
    if explicit_s1 and explicit_s2 and explicit_s3:
        return explicit_s1, explicit_s2, explicit_s3

    files = sorted(
        [
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".xls", ".xlsx", ".xlsm", ".csv"}
        ]
    )
    if not files:
        raise FileNotFoundError(f"No table files found under: {input_dir}")

    def pick(candidates: list[Path], patterns: tuple[str, ...]) -> Path | None:
        for pat in patterns:
            for f in candidates:
                if pat.lower() in f.name.lower():
                    return f
        return None

    s1 = explicit_s1 or pick(files, ("moesm2", "table_s1", "tables1", "s1"))
    s2 = explicit_s2 or pick(files, ("moesm3", "table_s2", "tables2", "s2"))
    s3 = explicit_s3 or pick(files, ("moesm4", "table_s3", "tables3", "s3"))
    missing = [name for name, val in (("S1", s1), ("S2", s2), ("S3", s3)) if val is None]
    if missing:
        raise FileNotFoundError(
            f"Could not identify {', '.join(missing)} files in {input_dir}. "
            "Pass --s1/--s2/--s3 explicitly."
        )
    return s1, s2, s3


def prepare_s1(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("S1 table is empty")
    header = make_unique_columns([str(x).strip() if x is not None else "" for x in df.iloc[0].tolist()])
    body = df.iloc[1:].copy().reset_index(drop=True)
    body.columns = header
    body["virus_row_index"] = np.arange(len(body), dtype=int)
    id_col = "ID" if "ID" in body.columns else body.columns[0]
    body["virus_id"] = body[id_col].astype(str).str.strip()
    missing_mask = body["virus_id"].eq("") | body["virus_id"].eq("None") | body["virus_id"].isna()
    body.loc[missing_mask, "virus_id"] = "virus_" + body.loc[missing_mask, "virus_row_index"].astype(str)
    if "Array" not in body.columns:
        body["Array"] = "unknown"
    return body


def prepare_s2(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("S2 table is empty")

    header_like = False
    if df.shape[1] >= 2:
        c0 = str(df.iloc[0, 0]).strip().lower()
        c1 = str(df.iloc[0, 1]).strip().lower()
        header_like = ("id" in c0 and ("glycan" in c1 or "iupac" in c1 or "structure" in c1)) or (
            "glycan" in c0 and ("iupac" in c1 or "structure" in c1)
        )

    if header_like:
        headers = make_unique_columns(
            [str(x).strip() if x is not None else "" for x in df.iloc[0].tolist()]
        )
        s2 = df.iloc[1:].copy().reset_index(drop=True)
        s2.columns = headers
        id_candidates = [c for c in s2.columns if c.lower() in {"id", "glycan_id"} or "id" == c.lower()]
        struct_candidates = [
            c
            for c in s2.columns
            if any(k in c.lower() for k in ("glycan", "iupac", "structure", "sequence"))
        ]
        id_col = id_candidates[0] if id_candidates else s2.columns[0]
        struct_col = struct_candidates[0] if struct_candidates else s2.columns[1]
    else:
        s2 = df.copy().reset_index(drop=True)
        id_col = s2.columns[0]
        struct_col = s2.columns[1] if s2.shape[1] > 1 else s2.columns[0]

    out = pd.DataFrame()
    ids = pd.to_numeric(s2[id_col], errors="coerce")
    fallback_ids = pd.Series(np.arange(1, len(ids) + 1), index=ids.index, dtype=float)
    out["glycan_id"] = ids.where(ids.notna(), fallback_ids).astype(int)
    out["structure_raw"] = s2[struct_col].astype(str).fillna("")
    out["structure_raw"] = out["structure_raw"].str.strip()
    out["s2_row_index"] = np.arange(len(out), dtype=int)
    out = out.drop_duplicates(subset=["glycan_id"], keep="first").reset_index(drop=True)
    return out


def prepare_s3(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("S3 table is empty")
    out = df.copy().reset_index(drop=True)
    out.columns = list(range(out.shape[1]))
    return out


def build_column_mapping(
    s2: pd.DataFrame,
    s3_ncols: int,
    mapping_mode: str,
    warnings: list[str],
) -> pd.DataFrame:
    mapping = pd.DataFrame({"s3_col_index": np.arange(s3_ncols, dtype=int)})
    mapping["slot_id"] = mapping["s3_col_index"].apply(lambda x: f"slot_{x + 1:03d}")
    mapping["glycan_id"] = pd.NA
    mapping["mapping_method"] = "unmapped"
    mapping["is_mapped"] = False

    if s3_ncols == len(s2):
        mapping["glycan_id"] = s2["glycan_id"].astype(int).values
        mapping["mapping_method"] = "direct_by_position"
        mapping["is_mapped"] = True
        return mapping

    if mapping_mode == "strict":
        raise ValueError(
            "S3 column count does not match S2 glycan count in strict mode: "
            f"S3={s3_ncols}, S2={len(s2)}"
        )

    n = min(s3_ncols, len(s2))
    mapping.loc[: n - 1, "glycan_id"] = s2["glycan_id"].astype(int).iloc[:n].values
    mapping.loc[: n - 1, "mapping_method"] = "positional_fallback"
    mapping.loc[: n - 1, "is_mapped"] = True

    warnings.append(
        "S3 columns and S2 glycans differ in size. Applied positional fallback mapping "
        f"for first {n} columns (S3={s3_ncols}, S2={len(s2)})."
    )
    if s3_ncols > len(s2):
        warnings.append(
            f"{s3_ncols - len(s2)} S3 columns remained unmapped and were excluded from training triples."
        )
    return mapping


def canonicalize_structure_string(raw: str) -> str:
    text = str(raw).strip()
    if not text:
        return ""
    text = text.replace('"', "").replace("'", "")
    text = re.sub(r"\s+", "", text)
    text = text.replace("SpSp", "Sp")
    return text


def strip_spacer_suffix(structure: str) -> str:
    return SPACER_SUFFIX_RE.sub("", structure).strip("-")


def split_linear_segment(segment: str) -> list[str]:
    if not segment:
        return []
    seg = segment.strip("-")
    if not seg:
        return []
    cuts = [0]
    depth = 0
    for i in range(1, len(seg)):
        ch = seg[i]
        prev = seg[i - 1]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif depth == 0 and ch.isupper() and (prev.isdigit() or prev == ")"):
            cuts.append(i)
    cuts.append(len(seg))
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        token = seg[a:b].strip("-")
        if token:
            out.append(token)
    return out


def is_branch_content(content: str) -> bool:
    return bool(BRANCH_LINKAGE_RE.search(content))


def tokenize_top_level(sequence: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    buf: list[str] = []
    i = 0
    while i < len(sequence):
        ch = sequence[i]
        if ch != "(":
            buf.append(ch)
            i += 1
            continue
        depth = 1
        j = i + 1
        while j < len(sequence) and depth > 0:
            if sequence[j] == "(":
                depth += 1
            elif sequence[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            # Unbalanced; treat rest as linear text.
            buf.append(sequence[i:])
            break
        content = sequence[i + 1 : j - 1]
        if is_branch_content(content):
            if buf:
                for tok in split_linear_segment("".join(buf)):
                    items.append(("residue", tok))
                buf = []
            items.append(("branch", content))
        else:
            # Modification parenthesis, keep with current residue token.
            buf.append("(" + content + ")")
        i = j
    if buf:
        for tok in split_linear_segment("".join(buf)):
            items.append(("residue", tok))
    return items


def split_residue_and_linkage(token: str) -> tuple[str, str]:
    m = LINKAGE_SUFFIX_RE.search(token)
    if not m:
        return token, ""
    linkage = m.group(1)
    residue = token[: m.start(1)]
    return residue, linkage


def parse_glycan_topology(structure: str) -> dict[str, Any] | None:
    clean = strip_spacer_suffix(canonicalize_structure_string(structure))
    if not clean:
        return None

    nodes: list[dict[str, Any]] = []

    def parse_seq(seq: str) -> int | None:
        items = tokenize_top_level(seq)
        pending_branch_roots: list[int] = []
        prev_main: int | None = None
        for kind, value in items:
            if kind == "branch":
                branch_root = parse_seq(value)
                if branch_root is not None:
                    pending_branch_roots.append(branch_root)
                continue
            residue, linkage = split_residue_and_linkage(value)
            node_id = len(nodes)
            nodes.append(
                {
                    "token": value,
                    "residue": residue,
                    "linkage": linkage,
                    "parent": None,
                }
            )
            if prev_main is not None:
                nodes[prev_main]["parent"] = node_id
            for br_root in pending_branch_roots:
                nodes[br_root]["parent"] = node_id
            pending_branch_roots = []
            prev_main = node_id
        return prev_main

    root = parse_seq(clean)
    if root is None or not nodes:
        return None

    children: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for node_id, node in enumerate(nodes):
        parent = node["parent"]
        if parent is not None and 0 <= parent < len(nodes):
            children[parent].append(node_id)

    depth = [0 for _ in nodes]
    stack = [(root, 0)]
    seen: set[int] = set()
    while stack:
        node_id, d = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        depth[node_id] = d
        for child in children.get(node_id, []):
            stack.append((child, d + 1))

    terminal_flags = [len(children[i]) == 0 for i in range(len(nodes))]

    terminal_distance = [0 for _ in nodes]

    def dist(node_id: int) -> int:
        if not children[node_id]:
            terminal_distance[node_id] = 0
            return 0
        v = 1 + min(dist(c) for c in children[node_id])
        terminal_distance[node_id] = v
        return v

    dist(root)

    linkage_flags = [bool(nodes[i]["linkage"]) for i in range(len(nodes))]
    tokens = [nodes[i]["token"] for i in range(len(nodes))]

    return {
        "tokens": tokens,
        "depth": depth,
        "terminal_flags": terminal_flags,
        "linkage_flags": linkage_flags,
        "terminal_distance": terminal_distance,
    }


def add_normalized_targets(df: pd.DataFrame, eps: float, robust: bool) -> pd.DataFrame:
    out = df.copy()
    out["y_raw"] = pd.to_numeric(out["y_raw"], errors="coerce")

    def log_transform(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().any() and s.dropna().min() >= 0:
            return np.log1p(s)
        return s

    out["y_log"] = out.groupby("virus_id")["y_raw"].transform(log_transform)

    if robust:
        center = out.groupby("virus_id")["y_log"].transform("median")
        q75 = out.groupby("virus_id")["y_log"].transform(lambda s: s.quantile(0.75))
        q25 = out.groupby("virus_id")["y_log"].transform(lambda s: s.quantile(0.25))
        denom = q75 - q25
        std = out.groupby("virus_id")["y_log"].transform(lambda s: s.std(ddof=0))
        denom = denom.where((denom.notna()) & (denom > eps), std)
    else:
        center = out.groupby("virus_id")["y_log"].transform("mean")
        denom = out.groupby("virus_id")["y_log"].transform(lambda s: s.std(ddof=0))

    denom = denom.where((denom.notna()) & (denom > eps), 1.0)
    out["y_norm"] = (out["y_log"] - center) / denom
    return out


def add_classification_labels(
    df: pd.DataFrame,
    binder_threshold: float,
    topk_fraction: float,
    strong_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    out["binder_binary"] = (out["y_norm"] > binder_threshold) & out["y_norm"].notna()
    out["binder_topk"] = False
    valid = out["y_norm"].notna()
    if valid.any():
        grouped = out.loc[valid].groupby("virus_id")["y_norm"]
        counts = grouped.transform("count")
        ranks = grouped.rank(method="first", ascending=False)
        k = np.ceil(counts * topk_fraction)
        out.loc[valid, "binder_topk"] = ranks <= k

    conditions = [
        out["y_norm"].isna(),
        out["y_norm"] >= strong_threshold,
        out["y_norm"] >= binder_threshold,
    ]
    choices = ["missing", "strong", "weak"]
    out["binder_strength"] = np.select(conditions, choices, default="non_binder")
    return out


def add_splits(
    df: pd.DataFrame,
    glycans: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    virus_holdout_frac: float,
) -> pd.DataFrame:
    out = df.copy()

    def random_split_for_pair(virus_id: str, glycan_id: int) -> str:
        p = stable_hash_fraction(f"{virus_id}|{glycan_id}")
        if p < train_frac:
            return "train"
        if p < train_frac + val_frac:
            return "val"
        return "test"

    out["split_random"] = [
        random_split_for_pair(str(v), int(g)) for v, g in zip(out["virus_id"], out["glycan_id"])
    ]

    alpha26_ids = set(
        glycans.loc[
            glycans["structure_string"].str.contains("a2-6", case=False, na=False)
            & glycans["structure_string"].str.contains("Neu", case=False, na=False),
            "glycan_id",
        ].astype(int)
    )

    def structural_split_for_pair(virus_id: str, glycan_id: int) -> str:
        if int(glycan_id) in alpha26_ids:
            return "test_alpha26"
        p = stable_hash_fraction(f"struct|{virus_id}|{glycan_id}")
        return "train" if p < 0.9 else "val"

    out["split_structural"] = [
        structural_split_for_pair(str(v), int(g)) for v, g in zip(out["virus_id"], out["glycan_id"])
    ]

    held_out_viruses = {
        str(v)
        for v in out["virus_id"].unique()
        if stable_hash_fraction(f"virus_holdout|{v}") < virus_holdout_frac
    }

    def virus_split_for_pair(virus_id: str, glycan_id: int) -> str:
        if str(virus_id) in held_out_viruses:
            return "test_virus"
        p = stable_hash_fraction(f"virus_train_val|{virus_id}|{glycan_id}")
        return "train" if p < 0.9 else "val"

    out["split_virus"] = [
        virus_split_for_pair(str(v), int(g)) for v, g in zip(out["virus_id"], out["glycan_id"])
    ]
    return out


def build_artifacts(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    out_dir: Path,
    missingness_threshold: float,
    binder_threshold: float,
    topk_fraction: float,
    strong_threshold: float,
    robust_norm: bool,
    norm_eps: float,
    mapping_mode: str,
    split_train_frac: float,
    split_val_frac: float,
    virus_holdout_frac: float,
) -> ParseArtifacts:
    warnings: list[str] = []

    s1_raw = load_first_table(s1_path)
    s2_raw = load_first_table(s2_path)
    s3_raw = load_first_table(s3_path)

    s1 = prepare_s1(s1_raw)
    s2 = prepare_s2(s2_raw)
    s3 = prepare_s3(s3_raw)

    if len(s1) != len(s3):
        warnings.append(
            f"S1/S3 row mismatch (S1={len(s1)}, S3={len(s3)}). Using min row count alignment."
        )
        n = min(len(s1), len(s3))
        s1 = s1.iloc[:n].reset_index(drop=True)
        s3 = s3.iloc[:n].reset_index(drop=True)
        s1["virus_row_index"] = np.arange(n, dtype=int)

    mapping = build_column_mapping(s2, s3.shape[1], mapping_mode, warnings)

    y_matrix = s3.apply(pd.to_numeric, errors="coerce")
    long_df = (
        y_matrix.reset_index(names="virus_row_index")
        .melt(id_vars="virus_row_index", var_name="s3_col_index", value_name="y_raw")
    )
    long_df["virus_row_index"] = long_df["virus_row_index"].astype(int)
    long_df["s3_col_index"] = long_df["s3_col_index"].astype(int)

    long_df = long_df.merge(
        s1[["virus_row_index", "virus_id", "Array"]],
        on="virus_row_index",
        how="left",
    )
    long_df = long_df.merge(
        mapping[["s3_col_index", "slot_id", "glycan_id", "mapping_method", "is_mapped"]],
        on="s3_col_index",
        how="left",
    )

    long_df = long_df[long_df["is_mapped"].fillna(False)].copy()
    long_df["glycan_id"] = pd.to_numeric(long_df["glycan_id"], errors="coerce").astype("Int64")
    long_df = long_df[long_df["glycan_id"].notna()].copy()
    long_df["glycan_id"] = long_df["glycan_id"].astype(int)

    virus_missing = long_df.groupby("virus_id")["y_raw"].apply(lambda s: s.isna().mean())
    glycan_missing = long_df.groupby("glycan_id")["y_raw"].apply(lambda s: s.isna().mean())
    keep_viruses = set(virus_missing[virus_missing <= missingness_threshold].index)
    keep_glycans = set(glycan_missing[glycan_missing <= missingness_threshold].index.astype(int))

    long_df = long_df[
        long_df["virus_id"].isin(keep_viruses) & long_df["glycan_id"].isin(keep_glycans)
    ].copy()

    long_df = add_normalized_targets(long_df, eps=norm_eps, robust=robust_norm)
    long_df = add_classification_labels(
        long_df,
        binder_threshold=binder_threshold,
        topk_fraction=topk_fraction,
        strong_threshold=strong_threshold,
    )

    s2["structure_string"] = s2["structure_raw"].apply(canonicalize_structure_string)
    s2["provenance_source"] = "Table_S2"
    s2["provenance_file"] = str(s2_path)

    used_glycan_ids = set(long_df["glycan_id"].astype(int).unique())
    glycans = s2[s2["glycan_id"].isin(used_glycan_ids)].copy()
    if glycans.empty:
        raise ValueError("No mapped glycans from S3 were found in S2 after filtering.")

    glycans = glycans.merge(
        mapping[["glycan_id", "mapping_method"]].dropna().drop_duplicates(),
        on="glycan_id",
        how="left",
    )
    glycans["mapping_method"] = glycans["mapping_method"].fillna("from_s2")

    token_rows: list[dict[str, Any]] = []
    for row in glycans.itertuples(index=False):
        parsed = parse_glycan_topology(row.structure_string)
        if parsed is None:
            token_rows.append(
                {
                    "glycan_id": int(row.glycan_id),
                    "structure_string": row.structure_string,
                    "parseable": False,
                    "tokens": json.dumps([]),
                    "depth": json.dumps([]),
                    "terminal_flags": json.dumps([]),
                    "linkage_flags": json.dumps([]),
                    "terminal_distance": json.dumps([]),
                    "parse_error": "unparsed",
                }
            )
            continue
        token_rows.append(
            {
                "glycan_id": int(row.glycan_id),
                "structure_string": row.structure_string,
                "parseable": True,
                "tokens": json.dumps(parsed["tokens"], separators=(",", ":")),
                "depth": json.dumps(parsed["depth"], separators=(",", ":")),
                "terminal_flags": json.dumps(parsed["terminal_flags"], separators=(",", ":")),
                "linkage_flags": json.dumps(parsed["linkage_flags"], separators=(",", ":")),
                "terminal_distance": json.dumps(
                    parsed["terminal_distance"], separators=(",", ":")
                ),
                "parse_error": "",
            }
        )

    tokenized = pd.DataFrame(token_rows)
    glycans = glycans.merge(tokenized[["glycan_id", "parseable"]], on="glycan_id", how="left")

    long_df = add_splits(
        long_df,
        glycans=glycans,
        train_frac=split_train_frac,
        val_frac=split_val_frac,
        virus_holdout_frac=virus_holdout_frac,
    )

    report = {
        "inputs": {
            "s1": str(s1_path),
            "s2": str(s2_path),
            "s3": str(s3_path),
        },
        "dimensions": {
            "s1_rows": int(len(s1)),
            "s2_rows": int(len(s2)),
            "s3_shape": [int(s3.shape[0]), int(s3.shape[1])],
        },
        "mapping": {
            "mapped_columns": int(mapping["is_mapped"].sum()),
            "total_s3_columns": int(len(mapping)),
            "mapped_glycans_used": int(glycans["glycan_id"].nunique()),
            "mapping_mode": mapping_mode,
        },
        "targets": {
            "triples_total": int(len(long_df)),
            "viruses_after_filter": int(long_df["virus_id"].nunique()),
            "glycans_after_filter": int(long_df["glycan_id"].nunique()),
            "missingness_threshold": missingness_threshold,
            "binder_threshold": binder_threshold,
            "topk_fraction": topk_fraction,
            "strong_threshold": strong_threshold,
        },
        "structure_coverage": {
            "parseable_glycans": int(tokenized["parseable"].sum()),
            "total_glycans": int(len(tokenized)),
            "parseable_fraction": float(tokenized["parseable"].mean()) if len(tokenized) else 0.0,
        },
        "warnings": warnings,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out_dir / "viral_glycan_bind_long.parquet", index=False)
    glycans.to_parquet(out_dir / "glycans.parquet", index=False)
    tokenized.to_parquet(out_dir / "glycans_tokenized.parquet", index=False)
    mapping.to_parquet(out_dir / "glycan_column_mapping.parquet", index=False)
    (out_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return ParseArtifacts(
        long_df=long_df,
        glycans_df=glycans,
        tokenized_df=tokenized,
        mapping_df=mapping,
        report=report,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=None, help="Optional supplementary ZIP to extract.")
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("./extracted_supplementary"),
        help="Extraction directory used when --zip is provided.",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing S1/S2/S3.")
    parser.add_argument("--s1", type=Path, default=None, help="Explicit path to Table S1 file.")
    parser.add_argument("--s2", type=Path, default=None, help="Explicit path to Table S2 file.")
    parser.add_argument("--s3", type=Path, default=None, help="Explicit path to Table S3 file.")
    parser.add_argument("--out-dir", type=Path, default=Path("./artifacts"), help="Output directory.")

    parser.add_argument(
        "--mapping-mode",
        choices=["auto", "strict", "positional_fallback"],
        default="auto",
        help="How to map S3 columns to S2 glycans when dimensions mismatch.",
    )
    parser.add_argument(
        "--missingness-threshold",
        type=float,
        default=0.5,
        help="Drop viruses/glycans with missingness above this fraction.",
    )
    parser.add_argument("--binder-threshold", type=float, default=1.0, help="y_norm threshold for binder_binary.")
    parser.add_argument(
        "--topk-fraction",
        type=float,
        default=0.10,
        help="Top-k fraction per virus for binder_topk (0-1).",
    )
    parser.add_argument(
        "--strong-threshold",
        type=float,
        default=2.0,
        help="y_norm threshold for binder_strength='strong'.",
    )
    parser.add_argument(
        "--norm-method",
        choices=["robust", "zscore"],
        default="robust",
        help="Row-wise normalization method.",
    )
    parser.add_argument("--norm-eps", type=float, default=1e-6, help="Stability epsilon for normalization.")
    parser.add_argument("--split-train-frac", type=float, default=0.8)
    parser.add_argument("--split-val-frac", type=float, default=0.1)
    parser.add_argument("--virus-holdout-frac", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    if args.zip is not None:
        args.extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "r") as zf:
            zf.extractall(args.extract_dir)
        input_dir = args.extract_dir

    s1_path, s2_path, s3_path = detect_table_paths(input_dir, args.s1, args.s2, args.s3)

    mode = args.mapping_mode
    if mode == "auto":
        mode = "positional_fallback"

    artifacts = build_artifacts(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        out_dir=args.out_dir,
        missingness_threshold=args.missingness_threshold,
        binder_threshold=args.binder_threshold,
        topk_fraction=args.topk_fraction,
        strong_threshold=args.strong_threshold,
        robust_norm=(args.norm_method == "robust"),
        norm_eps=args.norm_eps,
        mapping_mode=mode,
        split_train_frac=args.split_train_frac,
        split_val_frac=args.split_val_frac,
        virus_holdout_frac=args.virus_holdout_frac,
    )

    report = artifacts.report
    print("Preprocessing complete.")
    print(f"S1 rows: {report['dimensions']['s1_rows']}")
    print(f"S2 glycans: {report['dimensions']['s2_rows']}")
    print(f"S3 shape: {tuple(report['dimensions']['s3_shape'])}")
    print(f"Triples: {report['targets']['triples_total']}")
    print(
        "Parseable glycans used in training: "
        f"{report['structure_coverage']['parseable_glycans']}/"
        f"{report['structure_coverage']['total_glycans']}"
    )
    if report["warnings"]:
        print("Warnings:")
        for w in report["warnings"]:
            print(f"- {w}")


if __name__ == "__main__":
    main()
