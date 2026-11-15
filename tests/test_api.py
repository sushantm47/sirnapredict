"""API tests using FastAPI's TestClient (no server needed)."""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.features import reverse_complement

GUIDE = "UUGAUAUGUUGGAUGAUGG"
MRNA = ("AUGGUGAGCAAGGGCGAGGAGCUGUUCACCGGGGUGGUGCCCAUCCUGGUCGAGCUGGACGGCGACGUAAACGGC"
        "CACAAGUUCAGCGUGUCCGGCGAGGGCGAGGGCGAUGCCACCUACGGCAAGCUGACCCUGAAGUUCAUCUGCACC")


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_dir / "sirna_model.pkl"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pkl"))
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------- status

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy" and body["model_loaded"] is True and body["version"]


def test_model_info(client):
    info = client.get("/model/info").json()
    assert "GroupKFold" in info["validation"]
    assert info["best_model"] in {"ridge", "forest"}


def test_endpoints_return_503_without_a_model(client_without_model):
    assert client_without_model.get("/health").status_code == 503
    assert client_without_model.post("/predict", json={"guide": GUIDE}).status_code == 503
    assert client_without_model.post("/design", json={"mrna": MRNA}).status_code == 503


# --------------------------------------------------------------- predict

def test_predict_shape_and_consistency(client):
    body = client.post("/predict", json={"guide": GUIDE}).json()
    assert 0 <= body["predicted_knockdown"] <= 1
    assert body["sense"] == reverse_complement(GUIDE)
    assert body["call"] == ("active" if body["predicted_knockdown"] >= 0.7 else "inactive")
    assert body["rule_check"]["passed"] == sum(body["rule_check"]["rules"].values())


def test_predict_accepts_dna_and_lowercase(client):
    dna = GUIDE.replace("U", "T").lower()
    assert client.post("/predict", json={"guide": dna}).json()["guide"] == GUIDE


@pytest.mark.parametrize("guide", ["TOO SHORT", "UUGAUAUGUUGGAUGAUNN", "", GUIDE + "AA"])
def test_predict_rejects_bad_guides(client, guide):
    assert client.post("/predict", json={"guide": guide}).status_code == 422


def test_batch(client):
    guides = [GUIDE, reverse_complement(GUIDE)]
    body = client.post("/predict/batch", json={"guides": guides}).json()
    assert [b["guide"] for b in body] == guides


@pytest.mark.parametrize("guides", [[], ["UUGAUAUGUUGGAUGAUGG"] * 501])
def test_batch_size_limits(client, guides):
    assert client.post("/predict/batch", json={"guides": guides}).status_code == 422


def test_batch_reports_which_guide_is_bad(client):
    r = client.post("/predict/batch", json={"guides": [GUIDE, "NOTVALID"]})
    assert r.status_code == 422 and "guide 2" in r.json()["detail"]


# --------------------------------------------------------------- design

def test_design_ranks_candidates(client):
    body = client.post("/design", json={"mrna": MRNA, "top_n": 5}).json()
    assert body["transcript_length"] == len(MRNA)
    assert body["windows_scanned"] == len(MRNA) - 18
    assert body["windows_passing_filters"] <= body["windows_scanned"]

    candidates = body["candidates"]
    assert len(candidates) <= 5
    scores = [c["predicted_knockdown"] for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))
    for c in candidates:
        start = c["position"] - 1
        assert MRNA[start:start + 19] == c["target_site"]
        assert c["guide"] == reverse_complement(c["target_site"])
        assert 0.30 <= c["gc_content"] <= 0.52


def test_design_filters_can_exclude_everything(client):
    body = client.post("/design", json={"mrna": MRNA, "min_gc": 0.99, "max_gc": 1.0}).json()
    assert body["candidates"] == [] and body["windows_passing_filters"] == 0


def test_design_rejects_impossible_gc_range(client):
    assert client.post("/design", json={"mrna": MRNA, "min_gc": 0.8, "max_gc": 0.2}).status_code == 422


@pytest.mark.parametrize("mrna", ["AUGGCC", "AUGGCCNNNN" * 5])
def test_design_rejects_bad_transcripts(client, mrna):
    assert client.post("/design", json={"mrna": mrna}).status_code == 422


def test_design_profile_covers_every_window(client):
    body = client.post("/design/profile", json={"mrna": MRNA}).json()
    assert len(body["positions"]) == len(MRNA) - 18
    assert body["positions"][0] == 1
    assert all(0 <= s <= 1 for s in body["predicted_knockdown"])
