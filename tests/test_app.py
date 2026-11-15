"""Tests for the dashboard: helpers, API client, and the Streamlit pages."""

from pathlib import Path

import pytest
import requests

import api_client
from helpers import (api_error_message, candidates_frame, parse_fasta, parse_guide_list,
                     predictions_frame, rule_label)

APP = Path(__file__).resolve().parents[1] / "app"
GUIDE = "UUGAUAUGUUGGAUGAUGG"
PREDICTION = {
    "guide": GUIDE, "sense": "CCAUCAUCCAACAUAUCAA", "predicted_knockdown": 0.82, "call": "active",
    "properties": {"gc_content": 0.368, "dg_total": -29.41, "end_asymmetry": 3.23},
    "rule_check": {"passed": 6, "total": 7, "rules": {"gc_30_52": True, "a_at_sense_19": True,
                                                      "a_at_sense_3": True, "u_at_sense_10": False,
                                                      "no_gc_at_sense_19": True,
                                                      "no_g_at_sense_13": True,
                                                      "au_rich_guide_5p": True}},
}
DESIGN = {
    "transcript_length": 150, "windows_scanned": 132, "windows_passing_filters": 40,
    "candidates": [{"rank": 1, "position": 42, "target_site": "CCAUCAUCCAACAUAUCAA",
                    "guide": GUIDE, "predicted_knockdown": 0.88, "gc_content": 0.42,
                    "end_asymmetry": 2.1, "rules_passed": 6}],
}


# --------------------------------------------------------------- FASTA

def test_parse_fasta_reads_header_and_joins_lines():
    header, seq = parse_fasta(">NM_000546 TP53\nAUGGCC\nUUGAUAUGUUGGAUGAUGG\n")
    assert header == "NM_000546 TP53"
    assert seq == "AUGGCCUUGAUAUGUUGGAUGAUGG"


def test_parse_fasta_accepts_dna_and_stops_at_second_record():
    _, seq = parse_fasta(">a\natggccttgatatgttggatgatgg\n>b\nGGGG\n")
    assert seq == "AUGGCCUUGAUAUGUUGGAUGAUGG"


@pytest.mark.parametrize("text, message", [
    ("", "empty"),
    (">only a header\n", "No sequence"),
    ("AUGGCCUUGAUAUGUUGGAUGAUXX", "invalid letters"),
    ("AUGGCC", "at least 19"),
])
def test_parse_fasta_rejects_bad_input(text, message):
    with pytest.raises(ValueError, match=message):
        parse_fasta(text)


# --------------------------------------------------------------- guide lists

def test_parse_guide_list_keeps_good_and_reports_bad():
    guides, errors = parse_guide_list(f"{GUIDE}\nAUGGCC\n>header\n\n{GUIDE.lower()}\nXXXX")
    assert guides == [GUIDE, GUIDE]
    assert len(errors) == 2 and "line 2" in errors[0]


def test_parse_guide_list_accepts_commas():
    guides, errors = parse_guide_list(f"{GUIDE}, {GUIDE}")
    assert len(guides) == 2 and not errors


# --------------------------------------------------------------- tables

def test_candidates_frame_columns_and_percentages():
    table = candidates_frame(DESIGN["candidates"])
    assert table["predicted_knockdown_percent"].iloc[0] == 88.0
    assert table["gc_percent"].iloc[0] == 42.0
    assert "gc_content" not in table.columns


def test_candidates_frame_handles_no_results():
    assert candidates_frame([]).empty


def test_predictions_frame_flattens_nested_fields():
    table = predictions_frame([PREDICTION])
    assert table["rules_passed"].iloc[0] == 6
    assert table["gc_percent"].iloc[0] == 36.8


def test_rule_label_is_readable():
    assert rule_label("gc_30_52").startswith("GC content")
    assert rule_label("unknown_rule") == "unknown rule"


# --------------------------------------------------------------- API client

class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code, self._body = status_code, body

    def json(self):
        return self._body


def offline(*args, **kwargs):
    raise requests.ConnectionError("refused")


def test_health_offline(monkeypatch):
    monkeypatch.setattr(api_client.requests, "get", offline)
    assert api_client.health() == {"online": False}


def test_health_reports_missing_model(monkeypatch):
    body = {"status": "unavailable", "model_loaded": False}
    monkeypatch.setattr(api_client.requests, "get", lambda *a, **k: FakeResponse(503, body))
    status = api_client.health()
    assert status["online"] is True and status["model_loaded"] is False


def test_design_posts_filters(monkeypatch):
    sent = {}

    def fake_request(method, url, **kwargs):
        sent.update(method=method, url=url, json=kwargs["json"])
        return FakeResponse(200, DESIGN)

    monkeypatch.setattr(api_client.requests, "request", fake_request)
    assert api_client.design("AUGG" * 10, top_n=5, min_gc=0.2)["candidates"][0]["rank"] == 1
    assert sent["json"]["top_n"] == 5 and sent["json"]["min_gc"] == 0.2
    assert sent["url"].endswith("/design")


def test_client_raises_readable_errors(monkeypatch):
    body = {"detail": [{"loc": ["body", "guide"], "msg": "must be 19 nucleotides long"}]}
    monkeypatch.setattr(api_client.requests, "request", lambda *a, **k: FakeResponse(422, body))
    with pytest.raises(api_client.ApiError, match="guide: must be 19 nucleotides"):
        api_client.predict("AUGG")


def test_client_reports_unreachable_api(monkeypatch):
    monkeypatch.setattr(api_client.requests, "request", offline)
    with pytest.raises(api_client.ApiError, match="Cannot reach the API"):
        api_client.predict(GUIDE)


def test_batch_is_chunked(monkeypatch):
    sizes = []

    def fake_request(method, url, **kwargs):
        n = len(kwargs["json"]["guides"])
        sizes.append(n)
        return FakeResponse(200, [PREDICTION] * n)

    monkeypatch.setattr(api_client.requests, "request", fake_request)
    assert len(api_client.predict_batch([GUIDE] * 1200)) == 1200
    assert sizes == [500, 500, 200]


def test_api_error_message_formats():
    assert api_error_message(503, {"detail": "Model not loaded"}) == "Model not loaded"
    assert api_error_message(500, None) == "Request failed with status 500"


# --------------------------------------------------------------- Streamlit pages

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def run(page):
    return AppTest.from_file(str(APP / page), default_timeout=60).run()


def online(monkeypatch):
    monkeypatch.setattr(api_client, "health",
                        lambda: {"online": True, "model_loaded": True, "version": "test"})


def test_dashboard_warns_when_backend_is_offline(monkeypatch):
    monkeypatch.setattr(api_client, "health", lambda: {"online": False})
    at = run("Dashboard.py")
    assert not at.exception and any("offline" in e.value for e in at.error)


def test_dashboard_designs_from_the_example(monkeypatch):
    online(monkeypatch)
    monkeypatch.setattr(api_client, "design", lambda *a, **k: DESIGN)
    monkeypatch.setattr(api_client, "design_profile", lambda m: {
        "positions": [1, 2], "predicted_knockdown": [0.5, 0.6]})
    at = run("Dashboard.py")
    at.radio(key="source").set_value("Example").run()
    at.button(key="design").click().run()
    assert not at.exception
    assert any("Best candidate" in s.value for s in at.success)


def test_dashboard_shows_api_errors(monkeypatch):
    online(monkeypatch)

    def failing(*args, **kwargs):
        raise api_client.ApiError(422, "mRNA must be at least 19 nucleotides long")

    monkeypatch.setattr(api_client, "design", failing)
    at = run("Dashboard.py")
    at.radio(key="source").set_value("Example").run()
    at.button(key="design").click().run()
    assert not at.exception
    assert any("at least 19" in e.value for e in at.error)


def test_check_page_scores_a_guide(monkeypatch):
    online(monkeypatch)
    monkeypatch.setattr(api_client, "predict", lambda guide: PREDICTION)
    at = run("pages/1_Check_siRNA.py")
    at.button(key="score_one").click().run()
    assert not at.exception
    assert at.metric[0].value == "82.0%"


def test_model_info_page(monkeypatch):
    info = {
        "version": "test", "trained_at": "2026-01-01T00:00:00+00:00", "best_model": "forest",
        "validation": "GroupKFold by target gene, 5 folds", "active_threshold": 0.7,
        "data": {"sirnas": 2160, "targets": 24, "mean_efficacy": 0.43, "active_fraction": 0.15,
                 "active_threshold": 0.7},
        "cv_scores": {"rules": {"pearson_r": 0.54, "spearman_r": 0.53, "mae": 2.8, "auc": 0.76},
                      "ridge": {"pearson_r": 0.71, "spearman_r": 0.71, "mae": 0.14, "auc": 0.87},
                      "forest": {"pearson_r": 0.79, "spearman_r": 0.78, "mae": 0.12, "auc": 0.89}},
        "top_features": [{"feature": "end_asymmetry", "weight": 0.12}],
        "per_target": [{"target": "GENE1", "sirnas": 90, "pearson_r": 0.7, "mean_efficacy": 0.4}],
    }
    monkeypatch.setattr(api_client, "model_info", lambda: info)
    at = run("pages/3_Model_Info.py")
    assert not at.exception
    assert at.metric[0].value == "0.790"


def test_assay_data_page_runs(model_dir, monkeypatch):
    """The page reads model/training_data.csv, which src.train writes."""
    import shutil
    target = APP.parent / "model" / "training_data.csv"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_dir / "training_data.csv", target)
    at = run("pages/2_Assay_Data.py")
    assert not at.exception
