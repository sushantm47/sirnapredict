"""Tests for dataset loading and the training pipeline."""

import json

import pandas as pd
import pytest

from src.data import dataset_summary, load_dataset
from src.features import reverse_complement

ROWS = [
    {"gene": "GENE1", "antisense": "UUGAUAUGUUGGAUGAUGG", "inhibition": 85.0},
    {"gene": "GENE1", "antisense": "GCAUCAUCCAACAUAUCAA", "inhibition": 40.0},
    {"gene": "GENE2", "antisense": "AUAUAUAUAUAUAUGCGCG", "inhibition": 12.0},
]


def write(tmp_path, rows, name="raw.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --------------------------------------------------------------- loading

def test_detects_columns_and_scales_percentages(tmp_path):
    df = load_dataset(write(tmp_path, ROWS))
    assert list(df.columns) == ["guide", "efficacy", "target"]
    assert df["efficacy"].tolist() == [0.85, 0.40, 0.12]
    assert df["target"].tolist() == ["GENE1", "GENE1", "GENE2"]


def test_accepts_fractions_unchanged(tmp_path):
    rows = [{**r, "inhibition": r["inhibition"] / 100} for r in ROWS]
    assert load_dataset(write(tmp_path, rows))["efficacy"].max() == pytest.approx(0.85)


def test_trims_21mers_to_19(tmp_path):
    rows = [{**r, "antisense": r["antisense"] + "UU"} for r in ROWS]
    df = load_dataset(write(tmp_path, rows))
    assert (df["guide"].str.len() == 19).all()
    assert df["guide"].iloc[0] == ROWS[0]["antisense"]


def test_derives_guide_from_sense_strand(tmp_path):
    rows = [{"gene": r["gene"], "sense": reverse_complement(r["antisense"]),
             "inhibition": r["inhibition"]} for r in ROWS]
    assert load_dataset(write(tmp_path, rows))["guide"].iloc[0] == ROWS[0]["antisense"]


def test_drops_unusable_rows_and_duplicates(tmp_path):
    rows = ROWS + [{"gene": "GENE3", "antisense": "NOTASEQUENCE", "inhibition": 50.0},
                   {"gene": "GENE1", "antisense": ROWS[0]["antisense"], "inhibition": 80.0}]
    assert len(load_dataset(write(tmp_path, rows))) == 3


def test_reads_tsv(tmp_path):
    path = tmp_path / "raw.tsv"
    pd.DataFrame(ROWS).to_csv(path, sep="\t", index=False)
    assert len(load_dataset(path)) == 3


def test_missing_file_explains_how_to_get_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        load_dataset(tmp_path / "nope.csv")


def test_no_sequence_column(tmp_path):
    with pytest.raises(ValueError, match="sequence column"):
        load_dataset(write(tmp_path, [{"a": 1, "b": 2}, {"a": 3, "b": 4}]))


def test_infers_targets_from_mrna_context(tmp_path):
    """Files with no gene column but an mRNA context window: recover the transcripts."""
    import numpy as np
    from src.data import infer_targets_from_context
    from src.features import reverse_complement as rc

    rng = np.random.default_rng(0)
    rows, expected = [], []
    for gene in range(3):
        transcript = "".join(rng.choice(list("ACGU"), 1200))
        for start in range(0, 1100, 25):          # densely tiled, like a real library
            site = transcript[start + 19:start + 38]
            rows.append({"siRNA": rc(site), "mRNA": transcript[start:start + 57],
                         "label": round(float(rng.uniform(0, 1)), 3)})
            expected.append(gene)

    df = load_dataset(write(tmp_path, rows))
    assert df["target"].nunique() == 3
    # siRNAs from one transcript must all land in the same group
    groups = pd.DataFrame({"gene": expected, "target": df["target"]})
    assert (groups.groupby("gene")["target"].nunique() == 1).all()

    labels = infer_targets_from_context([r["mRNA"] for r in rows])
    assert len(set(labels)) == 3


def test_context_column_is_not_used_as_the_target(tmp_path):
    """Every row has a different mRNA window, so it must not become the group label."""
    rows = [{"gene": r["gene"], "antisense": r["antisense"], "mRNA": "AUGC" * 14 + r["antisense"],
             "inhibition": r["inhibition"]} for r in ROWS]
    assert load_dataset(write(tmp_path, rows))["target"].tolist() == ["GENE1", "GENE1", "GENE2"]


def test_summary(tmp_path):
    summary = dataset_summary(load_dataset(write(tmp_path, ROWS)))
    assert summary["sirnas"] == 3 and summary["targets"] == 2
    assert summary["active_fraction"] == pytest.approx(1 / 3)


# --------------------------------------------------------------- training

def test_training_writes_model_and_metrics(model_dir):
    info = json.loads((model_dir / "metrics.json").read_text())
    assert (model_dir / "sirna_model.pkl").exists()
    assert (model_dir / "training_data.csv").exists()
    assert info["best_model"] in {"ridge", "forest"}
    assert "GroupKFold" in info["validation"]
    assert len(info["top_features"]) == 15


def test_learned_model_beats_the_rules_baseline(model_dir):
    scores = json.loads((model_dir / "metrics.json").read_text())["cv_scores"]
    assert scores[json.loads((model_dir / "metrics.json").read_text())["best_model"]]["pearson_r"] \
        > scores["rules"]["pearson_r"]


def test_scores_are_in_sensible_ranges(model_dir):
    scores = json.loads((model_dir / "metrics.json").read_text())["cv_scores"]
    for name, s in scores.items():
        assert -1 <= s["pearson_r"] <= 1, name
        assert s["auc"] is None or 0 <= s["auc"] <= 1, name
