"""
Generate a SYNTHETIC siRNA dataset so the app can be demonstrated without a download.

The sequences and efficacies are made up. Knockdown is simulated from the published
design rules (end asymmetry, GC content, position-specific bases) plus noise and a
per-gene offset, so the pipeline behaves realistically - but no number here is a
real measurement. For real results, download a published dataset and run
`python -m scripts.prepare_data` (see the README).

    python -m scripts.make_demo_data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import GUIDE_LENGTH, end_asymmetry, gc_content, reverse_complement

BASES = "ACGU"
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "synthetic_demo.csv"


def simulate_efficacy(guide, rng, gene_offset):
    """A stand-in for a knockdown assay, driven by the real design rules."""
    sense = reverse_complement(guide)
    gc = gc_content(guide)
    effect = (
        0.045 * end_asymmetry(guide)          # unstable 5' end favours loading
        - 2.4 * (gc - 0.42) ** 2 * 4          # mid-range GC is best
        + 0.07 * (sense[18] == "A")
        + 0.05 * (sense[2] == "A")
        + 0.04 * (sense[9] == "U")
        - 0.06 * (sense[12] == "G")
        - 0.05 * (guide[:4].count("G") + guide[:4].count("C")) / 4
    )
    return float(np.clip(0.55 + effect + gene_offset + rng.normal(0, 0.14), 0.0, 1.0))


def make_demo_dataset(n_genes=24, per_gene=90, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_genes):
        gene = f"DEMO{g + 1:02d}"
        gene_offset = rng.normal(0, 0.06)
        # a transcript per gene, so siRNAs for one gene share sequence context
        transcript = "".join(rng.choice(list(BASES), 2000, p=[0.27, 0.23, 0.23, 0.27]))
        starts = rng.choice(len(transcript) - GUIDE_LENGTH, per_gene, replace=False)
        for start in starts:
            site = transcript[start:start + GUIDE_LENGTH]
            guide = reverse_complement(site)
            rows.append({"gene": gene, "antisense": guide, "sense": site,
                         "inhibition": round(100 * simulate_efficacy(guide, rng, gene_offset), 1)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--genes", type=int, default=24)
    parser.add_argument("--per-gene", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = make_demo_dataset(args.genes, args.per_gene, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC siRNAs across {df['gene'].nunique()} genes to {args.out}")
    print("These are simulated, not measured. See the README for the real dataset.")


if __name__ == "__main__":
    main()
