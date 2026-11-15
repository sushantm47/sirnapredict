"""
Load a published siRNA efficacy dataset into one tidy table.

Published siRNA datasets all hold the same three things but never agree on column
names, so the loader looks for each one by a list of likely names and falls back
to guessing from the contents. The tidy result always has:

    guide      19-nt guide (antisense) strand, 5'->3', RNA letters
    efficacy   measured knockdown, 0-1 (1 = complete knockdown)
    target     the gene the siRNA was designed against (used to group the splits)
"""

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.features import GUIDE_LENGTH, SequenceError, clean_sequence, reverse_complement

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sirna_efficacy.csv"

GUIDE_NAMES = ["guide", "antisense", "sirna_antisense", "guide_strand", "as", "sirna"]
SENSE_NAMES = ["sense", "sense_strand", "sirna_sense", "passenger", "ss"]
EFFICACY_NAMES = ["efficacy", "inhibition", "knockdown", "activity", "potency", "label", "y"]
TARGET_NAMES = ["target", "gene", "gene_name", "target_gene", "refseq", "accession"]
CONTEXT_NAMES = ["mrna", "context", "mrna_context", "transcript", "target_region"]
BASES = set("ACGUT")


def _find_column(df, candidates):
    lookup = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    for key, original in lookup.items():
        if any(name in key for name in candidates):
            return original
    return None


def _looks_like_sequences(series):
    values = series.dropna().astype(str).head(50)
    if values.empty:
        return False
    return all(len(v.strip()) >= GUIDE_LENGTH and set(v.strip().upper()) <= BASES for v in values)


def _guess_sequence_column(df):
    for column in df.columns:
        if _looks_like_sequences(df[column]):
            return column
    return None


def _guess_efficacy_column(df, exclude):
    numeric = [c for c in df.columns
               if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    # the efficacy column has many distinct values and sits in 0-1 or 0-100
    scored = [(df[c].nunique(), c) for c in numeric
              if df[c].between(-20, 150).mean() > 0.95 and df[c].nunique() > 10]
    return max(scored)[1] if scored else None


def _normalize_efficacy(series):
    """Accept 0-1 or 0-100 and return 0-1, clipped."""
    values = pd.to_numeric(series, errors="coerce")
    if values.max() > 1.5:
        values = values / 100.0
    return values.clip(0.0, 1.0)


def _trim_guide(seq):
    """
    Cut a sequence down to a 19-nt guide.

    Huesken-style files store 21-mers with a 2-nt 3' overhang; the convention in
    the literature is to drop the overhang and keep the first 19 bases.
    """
    cleaned = clean_sequence(seq, length=None, name="sequence")
    if len(cleaned) < GUIDE_LENGTH:
        raise SequenceError(f"sequence shorter than {GUIDE_LENGTH} nt")
    return cleaned[:GUIDE_LENGTH]


def infer_targets_from_context(contexts, chain_k=18, attach_k=12, min_group=20):
    """
    Work out which siRNAs hit the same transcript, using only the mRNA context.

    Several published files give the sequence window around each target site but
    no gene name. Windows from one transcript overlap each other, so chaining
    windows that share a long exact substring recovers the transcripts:

      1. link any two windows sharing a `chain_k`-mer and take connected
         components - this is strict, so a gene can break into fragments where
         the library did not tile it densely;
      2. attach each small fragment to whichever large group it shares the most
         `attach_k`-mers with.

    Only small fragments are absorbed, never one large group into another, so
    distinct genes are not merged by a chance repeat. Returns a list of labels.
    """
    windows = [str(c).upper().replace("X", "").strip() for c in contexts]
    parent = list(range(len(windows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def kmers(seq, k):
        return {seq[i:i + k] for i in range(len(seq) - k + 1)}

    seen = {}
    for i, window in enumerate(windows):
        for kmer in kmers(window, chain_k):
            if kmer in seen:
                union(seen[kmer], i)
            else:
                seen[kmer] = i

    labels = [find(i) for i in range(len(windows))]
    sizes = Counter(labels)
    large = {g for g, n in sizes.items() if n >= min_group}

    if large:
        profiles = defaultdict(set)
        for i, group in enumerate(labels):
            if group in large:
                profiles[group] |= kmers(windows[i], attach_k)

        members = defaultdict(list)
        for i, group in enumerate(labels):
            members[group].append(i)

        remap = {}
        for group, rows in members.items():
            if group in large:
                continue
            fragment = set().union(*(kmers(windows[i], attach_k) for i in rows))
            best, best_overlap = None, 0
            for candidate, profile in profiles.items():
                overlap = len(fragment & profile)
                if overlap > best_overlap:
                    best, best_overlap = candidate, overlap
            if best is not None:
                remap[group] = best
        labels = [remap.get(g, g) for g in labels]

    order = {g: i for i, (g, _) in enumerate(Counter(labels).most_common())}
    return [f"target_{order[g] + 1:02d}" for g in labels]


def load_dataset(path=DATA_PATH, guide_column=None, efficacy_column=None,
                 target_column=None, sense_column=None, context_column=None):
    """
    Read a CSV or TSV of siRNA measurements and return the tidy table.

    Column names are detected automatically; pass them explicitly to override.
    Rows with unusable sequences or missing efficacy values are dropped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset at {path}. See the README for how to download one, "
            "then run `python -m scripts.prepare_data <file>`.")
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    raw = pd.read_csv(path, sep=separator)
    if raw.empty:
        raise ValueError(f"{path.name} has no rows")

    guide_column = guide_column or _find_column(raw, GUIDE_NAMES)
    sense_column = sense_column or _find_column(raw, SENSE_NAMES)
    if guide_column is None and sense_column is None:
        guide_column = _guess_sequence_column(raw)
    if guide_column is None and sense_column is None:
        raise ValueError(f"Could not find a sequence column in {list(raw.columns)}")

    efficacy_column = efficacy_column or _find_column(raw, EFFICACY_NAMES)
    if efficacy_column is None:
        efficacy_column = _guess_efficacy_column(raw, {guide_column, sense_column})
    if efficacy_column is None:
        raise ValueError(f"Could not find an efficacy column in {list(raw.columns)}")

    target_column = target_column or _find_column(raw, TARGET_NAMES)
    context_column = context_column or _find_column(raw, CONTEXT_NAMES)
    # a context column is only useful for grouping if it differs row to row
    if context_column is not None and raw[context_column].nunique() < 2:
        context_column = None

    rows = []
    for _, row in raw.iterrows():
        try:
            if guide_column is not None:
                guide = _trim_guide(row[guide_column])
            else:  # only a sense strand: the guide is its reverse complement
                guide = reverse_complement(_trim_guide(row[sense_column]))
        except SequenceError:
            continue
        rows.append({"guide": guide,
                     "efficacy_raw": row[efficacy_column],
                     "target": str(row[target_column]).strip() if target_column else None,
                     "context": row[context_column] if context_column else None})

    tidy = pd.DataFrame(rows)
    if tidy.empty:
        raise ValueError("No usable rows after cleaning: check the sequence column")
    tidy["efficacy"] = _normalize_efficacy(tidy.pop("efficacy_raw"))
    tidy = tidy.dropna(subset=["efficacy"])
    tidy = tidy.drop_duplicates(subset="guide").reset_index(drop=True)

    if target_column is None:
        if context_column is not None:
            tidy["target"] = infer_targets_from_context(tidy["context"])
        else:
            tidy["target"] = "unknown"
    return tidy[["guide", "efficacy", "target"]]


def dataset_summary(df, active_threshold=0.7):
    """Headline numbers for the data page and the training log."""
    return {
        "sirnas": int(len(df)),
        "targets": int(df["target"].nunique()),
        "mean_efficacy": float(df["efficacy"].mean()),
        "active_fraction": float((df["efficacy"] >= active_threshold).mean()),
        "active_threshold": float(active_threshold),
    }
