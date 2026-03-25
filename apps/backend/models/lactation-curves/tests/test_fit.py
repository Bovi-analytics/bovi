"""Tests for the /fit endpoint (lactation curve fitting)."""

import httpx
import pytest


def test_fit_wood_returns_305_predictions(api: httpx.Client, sample_data: dict):
    r = api.post("/fit", json={**sample_data, "model": "wood"})
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 305
    assert all(isinstance(v, float) for v in data["predictions"])


def test_fit_defaults(api: httpx.Client, sample_data: dict):
    """Only required fields -- all defaults should work."""
    r = api.post("/fit", json=sample_data)
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 305


@pytest.mark.parametrize(
    "model",
    [
        "wood",
        "wilmink",
        "ali_schaeffer",
        "fischer",
        "milkbot",
    ],
)
def test_fit_all_models(api: httpx.Client, sample_data: dict, model: str):
    r = api.post("/fit", json={**sample_data, "model": model})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) >= 305


def test_fit_missing_dim(api: httpx.Client):
    """Missing required field dim should return 422."""
    r = api.post("/fit", json={"milkrecordings": [15.0, 25.0, 30.0]})
    assert r.status_code == 422


def test_fit_missing_milkrecordings(api: httpx.Client):
    """Missing required field milkrecordings should return 422."""
    r = api.post("/fit", json={"dim": [10, 30, 60]})
    assert r.status_code == 422


def test_fit_empty_body(api: httpx.Client):
    """Empty request body should return 422."""
    r = api.post("/fit", json={})
    assert r.status_code == 422


def test_fit_invalid_model(api: httpx.Client, sample_data: dict):
    """Invalid model name should return 422."""
    r = api.post("/fit", json={**sample_data, "model": "nonexistent"})
    assert r.status_code == 422


def test_fit_mismatched_lengths(api: httpx.Client):
    """dim and milkrecordings with different lengths should return 422."""
    r = api.post(
        "/fit",
        json={
            "dim": [10, 30, 60, 90],
            "milkrecordings": [15.0, 25.0],
        },
    )
    assert r.status_code == 422


def test_fit_empty_lists(api: httpx.Client):
    """Empty dim and milkrecordings should return 422."""
    r = api.post("/fit", json={"dim": [], "milkrecordings": []})
    assert r.status_code == 422


def test_fit_single_point(api: httpx.Client):
    """Single data point is too few for fitting -- should return 422."""
    r = api.post("/fit", json={"dim": [10], "milkrecordings": [15.0]})
    assert r.status_code == 422


def test_fit_parity_zero(api: httpx.Client, sample_data: dict):
    """Parity 0 violates ge=1 constraint -- should return 422."""
    r = api.post("/fit", json={**sample_data, "parity": 0})
    assert r.status_code == 422


def test_fit_wrong_method(api: httpx.Client):
    """GET on a POST-only endpoint should return 405."""
    r = api.get("/fit")
    assert r.status_code == 405
