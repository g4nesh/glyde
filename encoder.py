""" glycan encoder (turns IUPAC tokens into fixed-length vectors). 
decoder step will be future work, but we can use the same architecture for 
both encoding and decoding, just with different projection matrices and training objectives
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EncoderConfig:
    embedding_dim: int = 128
    random_seed: int = 13
    unknown_token: str = "[UNK]"
    depth_weight: float = 0.35
    terminal_weight: float = 0.50
    linkage_weight: float = 0.20
    distance_weight: float = 0.25
    l2_normalize: bool = True


def _to_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, np.ndarray):
        return raw.tolist()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    if pd.isna(raw):
        return []
    return [raw]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pad_or_trim(values: list[Any], target_len: int, fill: Any) -> list[Any]:
    if len(values) < target_len:
        values = values + [fill] * (target_len - len(values))
    elif len(values) > target_len:
        values = values[:target_len]
    return values


class TopologyBiasedGlycanEncoder:
    def __init__(
        self,
        config: EncoderConfig,
        token_to_id: dict[str, int],
        token_embeddings: np.ndarray,
        topology_projection: np.ndarray,
    ) -> None:
        self.config = config
        self.token_to_id = token_to_id
        self.token_embeddings = token_embeddings
        self.topology_projection = topology_projection
        self.unknown_id = self.token_to_id[self.config.unknown_token]

    @classmethod
    def from_tokenized_dataframe(
        cls, tokenized_df: pd.DataFrame, config: EncoderConfig | None = None
    ) -> "TopologyBiasedGlycanEncoder":
        cfg = config or EncoderConfig()
        vocab: set[str] = set()
        for raw in tokenized_df.get("tokens", []):
            for tok in _to_list(raw):
                token = str(tok).strip()
                if token:
                    vocab.add(token)

        ordered_vocab = [cfg.unknown_token] + sorted(vocab)
        token_to_id = {token: idx for idx, token in enumerate(ordered_vocab)}

        rng = np.random.default_rng(cfg.random_seed)
        token_embeddings = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(cfg.embedding_dim),
            size=(len(ordered_vocab), cfg.embedding_dim),
        ).astype(np.float32)
        topology_projection = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(cfg.embedding_dim),
            size=(4, cfg.embedding_dim),
        ).astype(np.float32)

        return cls(
            config=cfg,
            token_to_id=token_to_id,
            token_embeddings=token_embeddings,
            topology_projection=topology_projection,
        )

    def _encode_from_components(
        self,
        tokens: list[str],
        depth: list[float],
        terminal_flags: list[bool],
        linkage_flags: list[bool],
        terminal_distance: list[float],
    ) -> np.ndarray:
        if not tokens:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

        token_ids = [self.token_to_id.get(tok, self.unknown_id) for tok in tokens]
        token_matrix = self.token_embeddings[token_ids]

        depth_arr = np.asarray(depth, dtype=np.float32)
        terminal_arr = np.asarray(terminal_flags, dtype=np.float32)
        linkage_arr = np.asarray(linkage_flags, dtype=np.float32)
        distance_arr = np.asarray(terminal_distance, dtype=np.float32)

        depth_norm = depth_arr / (np.max(depth_arr) + 1.0) if len(depth_arr) else depth_arr
        distance_norm = 1.0 / (1.0 + distance_arr)
        topology_features = np.stack(
            [depth_norm, terminal_arr, linkage_arr, distance_norm], axis=1
        ).astype(np.float32)

        topology_matrix = topology_features @ self.topology_projection
        token_plus_topology = token_matrix + topology_matrix

        weights = (
            1.0
            + self.config.depth_weight * (1.0 / (1.0 + depth_arr))
            + self.config.terminal_weight * terminal_arr
            + self.config.linkage_weight * linkage_arr
            + self.config.distance_weight * (1.0 / (1.0 + distance_arr))
        )
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            weights = np.ones_like(weights, dtype=np.float32) / len(weights)
        else:
            weights = (weights / weight_sum).astype(np.float32)

        embedding = (token_plus_topology * weights[:, None]).sum(axis=0)
        if self.config.l2_normalize:
            norm = float(np.linalg.norm(embedding))
            if norm > 0:
                embedding = embedding / norm
        return embedding.astype(np.float32)

    def encode_row(self, row: pd.Series) -> np.ndarray:
        tokens = [str(x) for x in _to_list(row.get("tokens")) if str(x).strip()]
        n = len(tokens)
        if n == 0:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

        depth = _pad_or_trim([_to_float(x, 0.0) for x in _to_list(row.get("depth"))], n, 0.0)
        terminal_flags = _pad_or_trim([_to_bool(x) for x in _to_list(row.get("terminal_flags"))], n, False)
        linkage_flags = _pad_or_trim([_to_bool(x) for x in _to_list(row.get("linkage_flags"))], n, False)
        terminal_distance = _pad_or_trim(
            [_to_float(x, 0.0) for x in _to_list(row.get("terminal_distance"))], n, 0.0
        )

        return self._encode_from_components(
            tokens=tokens,
            depth=[float(x) for x in depth],
            terminal_flags=[bool(x) for x in terminal_flags],
            linkage_flags=[bool(x) for x in linkage_flags],
            terminal_distance=[float(x) for x in terminal_distance],
        )

    def encode_dataframe(
        self,
        tokenized_df: pd.DataFrame,
        parseable_only: bool = True,
    ) -> pd.DataFrame:
        work_df = tokenized_df
        if parseable_only and "parseable" in tokenized_df.columns:
            work_df = tokenized_df[tokenized_df["parseable"].fillna(False)].copy()
        else:
            work_df = tokenized_df.copy()

        embeddings: list[np.ndarray] = []
        for _, row in work_df.iterrows():
            embeddings.append(self.encode_row(row))

        if embeddings:
            matrix = np.vstack(embeddings)
        else:
            matrix = np.empty((0, self.config.embedding_dim), dtype=np.float32)

        out = pd.DataFrame(
            matrix,
            columns=[f"enc_{i:03d}" for i in range(self.config.embedding_dim)],
        )
        if "glycan_id" in work_df.columns:
            out.insert(0, "glycan_id", work_df["glycan_id"].astype(int).values)
        if "parseable" in work_df.columns:
            out.insert(1 if "glycan_id" in out.columns else 0, "parseable", work_df["parseable"].values)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/glycans_tokenized.parquet"),
        help="Tokenized glycan parquet produced by prepare_binding_artifacts.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/glycans_encoded.parquet"),
        help="Where to write encoded vectors",
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--include-unparseable",
        action="store_true",
        help="Encode all rows, not only parseable glycans.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input tokenized parquet not found: {args.input}")

    tokenized_df = pd.read_parquet(args.input)
    config = EncoderConfig(embedding_dim=args.embedding_dim, random_seed=args.seed)
    encoder = TopologyBiasedGlycanEncoder.from_tokenized_dataframe(tokenized_df, config=config)
    encoded_df = encoder.encode_dataframe(
        tokenized_df, parseable_only=not args.include_unparseable
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded_df.to_parquet(args.output, index=False)

    print(f"Input rows: {len(tokenized_df)}")
    print(f"Encoded rows: {len(encoded_df)}")
    print(f"Embedding dim: {config.embedding_dim}")
    print(f"Vocab size: {len(encoder.token_to_id)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
