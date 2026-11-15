"""
Turn a downloaded siRNA dataset into the tidy file the rest of the project reads.

    python -m scripts.prepare_data path/to/downloaded.csv

Column names are detected automatically; override them if the guess is wrong:

    python -m scripts.prepare_data raw.csv --guide-column siRNA --efficacy-column inhibition
"""

import argparse
from pathlib import Path

from src.data import DATA_PATH, dataset_summary, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="the downloaded CSV or TSV")
    parser.add_argument("--out", type=Path, default=DATA_PATH)
    parser.add_argument("--guide-column")
    parser.add_argument("--sense-column")
    parser.add_argument("--efficacy-column")
    parser.add_argument("--target-column")
    args = parser.parse_args()

    df = load_dataset(args.source, guide_column=args.guide_column,
                      sense_column=args.sense_column,
                      efficacy_column=args.efficacy_column,
                      target_column=args.target_column)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    summary = dataset_summary(df)
    print(f"Wrote {summary['sirnas']} siRNAs across {summary['targets']} targets to {args.out}")
    print(f"Mean knockdown {summary['mean_efficacy']:.1%}, "
          f"{summary['active_fraction']:.1%} active at >={summary['active_threshold']:.0%}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
