"""
siRNA Design API.

Run from the project root:  uvicorn src.api:app --reload
Docs at http://127.0.0.1:8000/docs
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features import (GUIDE_LENGTH, SequenceError, clean_sequence, duplex_free_energy,
                          end_asymmetry, featurize_many, gc_content, reverse_complement,
                          reynolds_rules, reynolds_score, scan_transcript)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "model" / "sirna_model.pkl"
MAX_TRANSCRIPT = 20_000
MAX_GUIDES = 500


def model_path():
    return Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


def load_bundle(path=None):
    path = Path(path) if path else model_path()
    return joblib.load(path) if path.exists() else None


# ----------------------------------------------------------------- schemas

class GuideRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"guide": "UUGAUAUGUUGGAUGAUGG"}]})

    guide: str = Field(..., description=f"{GUIDE_LENGTH}-nt guide (antisense) strand, 5'->3'")

    @field_validator("guide")
    @classmethod
    def valid_guide(cls, value):
        try:
            return clean_sequence(value, name="guide")
        except SequenceError as e:
            raise ValueError(str(e)) from e


class GuideBatchRequest(BaseModel):
    guides: list[str] = Field(..., min_length=1, max_length=MAX_GUIDES)


class DesignRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "mrna": ("AUGGCCUUGAUAUGUUGGAUGAUGGAACCUUGGACCAUGGCUUAGCCAUGGUUACCGGAUC"
                 "CAUGGUUACCGGAUCCAUGGCUUAGCCAUGGUUACCGGAUCC"),
        "top_n": 10}]})

    mrna: str = Field(..., description="Target mRNA or CDS sequence, 5'->3'")
    top_n: int = Field(10, ge=1, le=100, description="How many candidates to return")
    min_gc: float = Field(0.30, ge=0.0, le=1.0)
    max_gc: float = Field(0.52, ge=0.0, le=1.0)
    avoid_runs: bool = Field(True, description="Reject candidates with 4+ identical bases in a row")

    @field_validator("mrna")
    @classmethod
    def valid_mrna(cls, value):
        try:
            cleaned = clean_sequence(value, length=None, name="mRNA")
        except SequenceError as e:
            raise ValueError(str(e)) from e
        if len(cleaned) < GUIDE_LENGTH:
            raise ValueError(f"mRNA must be at least {GUIDE_LENGTH} nucleotides long")
        if len(cleaned) > MAX_TRANSCRIPT:
            raise ValueError(f"mRNA must be at most {MAX_TRANSCRIPT:,} nucleotides long")
        return cleaned


class RuleCheck(BaseModel):
    passed: int
    total: int
    rules: dict[str, bool]


class Properties(BaseModel):
    gc_content: float
    dg_total: float = Field(..., description="Duplex free energy, kcal/mol (more negative = stabler)")
    end_asymmetry: float = Field(..., description="5' minus 3' stability; positive favours loading")


class GuidePrediction(BaseModel):
    guide: str
    sense: str
    predicted_knockdown: float = Field(..., ge=0, le=1)
    call: Literal["active", "inactive"] = Field(..., description="Active at >=70% predicted knockdown")
    properties: Properties
    rule_check: RuleCheck


class Candidate(BaseModel):
    rank: int
    position: int = Field(..., description="1-based start of the target site in the mRNA")
    target_site: str
    guide: str
    predicted_knockdown: float
    gc_content: float
    end_asymmetry: float
    rules_passed: int


class DesignResponse(BaseModel):
    transcript_length: int
    windows_scanned: int
    windows_passing_filters: int
    candidates: list[Candidate]


class Health(BaseModel):
    status: Literal["healthy", "unavailable"]
    model_loaded: bool
    version: Optional[str] = None


# ----------------------------------------------------------------- app

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle()
    if app.state.bundle is None:
        print(f"WARNING: no model at {model_path()}. Run `python -m src.train` first.")
    yield


app = FastAPI(
    title="siRNAPredict API",
    description=("Predict how well an siRNA will knock down its target, and scan a "
                 "transcript for the best candidates.\n\n"
                 "**Research and educational use only.** Not validated for therapeutic design "
                 "and not a substitute for experimental testing."),
    version="1.0.0",
    lifespan=lifespan,
)


def get_bundle(request: Request):
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Model not loaded. Run `python -m src.train` and restart the API.")
    return bundle


def predict_guides(bundle, guides):
    """Predicted knockdown (0-1) for a list of cleaned guides."""
    raw = bundle["model"].predict(featurize_many(guides))
    return np.clip(raw, 0.0, 1.0)


def describe(guide, knockdown):
    rules = reynolds_rules(guide)
    return {
        "guide": guide,
        "sense": reverse_complement(guide),
        "predicted_knockdown": round(float(knockdown), 4),
        "call": "active" if knockdown >= 0.7 else "inactive",
        "properties": {
            "gc_content": round(gc_content(guide), 4),
            "dg_total": round(duplex_free_energy(guide), 2),
            "end_asymmetry": round(end_asymmetry(guide), 2),
        },
        "rule_check": {"passed": reynolds_score(guide), "total": len(rules), "rules": rules},
    }


def has_long_run(seq, limit=4):
    run = 1
    for previous, current in zip(seq, seq[1:]):
        run = run + 1 if current == previous else 1
        if run >= limit:
            return True
    return False


@app.get("/health", response_model=Health, tags=["status"])
def health(request: Request, response: Response):
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Health(status="unavailable", model_loaded=False)
    return Health(status="healthy", model_loaded=True, version=bundle["info"]["version"])


@app.get("/model/info", tags=["status"])
def model_info(request: Request):
    """How the model was trained, how it was validated, and how well it scored."""
    return get_bundle(request)["info"]


@app.post("/predict", response_model=GuidePrediction, tags=["predictions"])
def predict(payload: GuideRequest, request: Request):
    """Score one guide strand and check it against the design rules."""
    bundle = get_bundle(request)
    knockdown = predict_guides(bundle, [payload.guide])[0]
    return describe(payload.guide, knockdown)


@app.post("/predict/batch", response_model=list[GuidePrediction], tags=["predictions"])
def predict_batch(payload: GuideBatchRequest, request: Request):
    """Score up to 500 guides at once, for example a list from a supplier."""
    bundle = get_bundle(request)
    try:
        guides = [clean_sequence(g, name=f"guide {i + 1}") for i, g in enumerate(payload.guides)]
    except SequenceError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    return [describe(g, k) for g, k in zip(guides, predict_guides(bundle, guides))]


@app.post("/design", response_model=DesignResponse, tags=["predictions"])
def design(payload: DesignRequest, request: Request):
    """Walk a target transcript and rank every possible siRNA against it."""
    bundle = get_bundle(request)
    if payload.min_gc > payload.max_gc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "min_gc must not be greater than max_gc")
    windows = scan_transcript(payload.mrna)
    kept = [w for w in windows
            if payload.min_gc <= gc_content(w["guide"]) <= payload.max_gc
            and not (payload.avoid_runs and has_long_run(w["guide"]))]
    if not kept:
        return {"transcript_length": len(payload.mrna), "windows_scanned": len(windows),
                "windows_passing_filters": 0, "candidates": []}

    scores = predict_guides(bundle, [w["guide"] for w in kept])
    ranked = sorted(zip(kept, scores), key=lambda t: -t[1])[:payload.top_n]
    candidates = [{
        "rank": i + 1,
        "position": w["position"],
        "target_site": w["target_site"],
        "guide": w["guide"],
        "predicted_knockdown": round(float(s), 4),
        "gc_content": round(gc_content(w["guide"]), 4),
        "end_asymmetry": round(end_asymmetry(w["guide"]), 2),
        "rules_passed": reynolds_score(w["guide"]),
    } for i, (w, s) in enumerate(ranked)]
    return {"transcript_length": len(payload.mrna), "windows_scanned": len(windows),
            "windows_passing_filters": len(kept), "candidates": candidates}


@app.post("/design/profile", tags=["predictions"])
def design_profile(payload: DesignRequest, request: Request):
    """Predicted knockdown at every position along the transcript, for plotting."""
    bundle = get_bundle(request)
    windows = scan_transcript(payload.mrna)
    scores = predict_guides(bundle, [w["guide"] for w in windows])
    return {"positions": [w["position"] for w in windows],
            "predicted_knockdown": [round(float(s), 4) for s in scores]}
