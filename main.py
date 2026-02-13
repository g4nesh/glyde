from pathlib import Path

import pandas as pd

from encoder import EncoderConfig, TopologyBiasedGlycanEncoder

def main() -> None:
    title = "Glyde: A domain-aware, topology-biased glycan language model for viral glycan binding prediction"
    thesis = "Viral glycan binding is a high-impact, tractable domain for a structure-aware glycan language model."

    print("Research Proposal")
    print(title)
    print()
    print("One-sentence thesis:")
    print(thesis)
    print()

    tokenized_path = Path("artifacts/glycans_tokenized.parquet")
    if not tokenized_path.exists():
        print("Encoder status:")
        print(f"- tokenized artifact missing at {tokenized_path}")
        return

    tokenized_df = pd.read_parquet(tokenized_path)
    encoder = TopologyBiasedGlycanEncoder.from_tokenized_dataframe(
        tokenized_df,
        config=EncoderConfig(embedding_dim=64, random_seed=13),
    )
    encoded_df = encoder.encode_dataframe(tokenized_df, parseable_only=True)
    print("Encoder status:")
    print(f"- tokenized glycans: {len(tokenized_df)}")
    print(f"- encoded glycans: {len(encoded_df)}")
    print(f"- encoder dim: {encoder.config.embedding_dim}")
    print(f"- encoder vocab size: {len(encoder.token_to_id)}")


if __name__ == "__main__":
    main()
