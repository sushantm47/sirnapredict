"""Client for the siRNA Design API."""

import os

import requests

from helpers import api_error_message

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 60
BATCH_LIMIT = 500


class ApiError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def _request(method, path, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as e:
        raise ApiError(None, f"Cannot reach the API at {API_URL}") from e
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code != 200:
        raise ApiError(r.status_code, api_error_message(r.status_code, body))
    return body


def health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
    except requests.RequestException:
        return {"online": False}
    try:
        body = r.json()
    except ValueError:
        body = {}
    return {**body, "online": True, "status_code": r.status_code}


def model_info():
    return _request("GET", "/model/info")


def predict(guide):
    return _request("POST", "/predict", json={"guide": guide})


def predict_batch(guides, on_progress=None):
    results = []
    for start in range(0, len(guides), BATCH_LIMIT):
        chunk = guides[start:start + BATCH_LIMIT]
        results.extend(_request("POST", "/predict/batch", json={"guides": chunk}))
        if on_progress:
            on_progress(len(results), len(guides))
    return results


def design(mrna, top_n=10, min_gc=0.30, max_gc=0.52, avoid_runs=True):
    return _request("POST", "/design", json={"mrna": mrna, "top_n": top_n, "min_gc": min_gc,
                                             "max_gc": max_gc, "avoid_runs": avoid_runs})


def design_profile(mrna):
    return _request("POST", "/design/profile", json={"mrna": mrna})
