"""Tests for the /predict endpoint (MilkBot direct evaluation)."""

import httpx


def test_predict_returns_predictions(api: httpx.Client):
    r = api.post(
        "/predict",
        json={
            "t": [1, 30, 60, 90, 120, 150, 200, 250, 305],
            "a": 40.0,
            "b": 20.0,
            "c": 0.5,
            "d": 0.003,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 9
    assert all(isinstance(v, float) for v in data["predictions"])


def test_predict_missing_field(api: httpx.Client):
    """Missing b, c, d should return 422."""
    r = api.post("/predict", json={"t": [1, 30], "a": 40.0})
    assert r.status_code == 422


def test_predict_empty_body(api: httpx.Client):
    """Empty request body should return 422."""
    r = api.post("/predict", json={})
    assert r.status_code == 422


def test_predict_invalid_t_type(api: httpx.Client):
    """Non-list t should return 422."""
    r = api.post(
        "/predict",
        json={
            "t": "not a list",
            "a": 40.0,
            "b": 20.0,
            "c": 0.5,
            "d": 0.003,
        },
    )
    assert r.status_code == 422


def test_predict_empty_t(api: httpx.Client):
    """Empty t list should still return 200 with empty predictions."""
    r = api.post(
        "/predict",
        json={
            "t": [],
            "a": 40.0,
            "b": 20.0,
            "c": 0.5,
            "d": 0.003,
        },
    )
    assert r.status_code == 200
    assert r.json()["predictions"] == []


def test_predict_wrong_method(api: httpx.Client):
    """GET on a POST-only endpoint should return 405."""
    r = api.get("/predict")
    assert r.status_code == 405
