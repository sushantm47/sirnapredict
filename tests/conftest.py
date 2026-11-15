"""Shared fixtures: a small synthetic dataset and a model trained on it."""

import pytest

from scripts.make_demo_data import make_demo_dataset
from src.train import train

GUIDE = "UUGAUAUGUUGGAUGAUGG"


@pytest.fixture(scope="session")
def dataset_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "sirna.csv"
    make_demo_dataset(n_genes=8, per_gene=60, seed=1).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def model_dir(dataset_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    train(data_path=dataset_path, model_dir=out, n_splits=3)
    return out
