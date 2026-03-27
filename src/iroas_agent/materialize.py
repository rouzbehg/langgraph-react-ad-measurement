from __future__ import annotations

import argparse
from pathlib import Path

from .data import generate_campaigns, write_campaign_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize synthetic campaign datasets to disk.")
    parser.add_argument("--count", type=int, default=100, help="Number of campaigns to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_campaigns.csv",
        help="Output path ending in .csv, .json, or .jsonl.",
    )
    args = parser.parse_args()

    campaigns = generate_campaigns(count=args.count, seed=args.seed)
    output_path = write_campaign_dataset(campaigns, Path(args.output))
    print(f"Wrote {args.count} campaigns to {output_path}")


if __name__ == "__main__":
    main()
