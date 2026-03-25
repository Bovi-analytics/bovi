"""Tests for the /characteristic endpoint."""

import httpx
import pytest


def test_characteristic_cumulative(api: httpx.Client, sample_data: dict):
    r = api.post(
        "/characteristic",
        json={
            **sample_data,
            "characteristic": "cumulative_milk_yield",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "value" in data
    assert isinstance(data["value"], float)
    assert data["value"] > 0


def test_characteristic_peak_yield(api: httpx.Client, sample_data: dict):
    r = api.post(
        "/characteristic",
        json={
            **sample_data,
            "characteristic": "peak_yield",
        },
    )
    assert r.status_code == 200
    assert r.json()["value"] > 0


def test_characteristic_time_to_peak(api: httpx.Client, sample_data: dict):
    r = api.post(
        "/characteristic",
        json={
            **sample_data,
            "characteristic": "time_to_peak",
        },
    )
    assert r.status_code == 200
    value = r.json()["value"]
    assert 0 < value < 305


def test_characteristic_persistency(api: httpx.Client, sample_data: dict):
    r = api.post(
        "/characteristic",
        json={
            **sample_data,
            "characteristic": "persistency",
        },
    )
    assert r.status_code == 200
    assert isinstance(r.json()["value"], float)


@pytest.mark.parametrize(
    "characteristic",
    [
        "time_to_peak",
        "peak_yield",
        "cumulative_milk_yield",
        "persistency",
    ],
)
def test_characteristic_all_types(
    api: httpx.Client,
    sample_data: dict,
    characteristic: str,
):
    r = api.post(
        "/characteristic",
        json={
            **sample_data,
            "characteristic": characteristic,
        },
    )
    assert r.status_code == 200
    assert isinstance(r.json()["value"], float)


def test_characteristic_defaults(api: httpx.Client, sample_data: dict):
    """Only required fields -- all defaults should work."""
    r = api.post("/characteristic", json=sample_data)
    assert r.status_code == 200
    assert isinstance(r.json()["value"], float)


def test_characteristic_missing_dim(api: httpx.Client):
    """Missing required field dim should return 422."""
    r = api.post("/characteristic", json={"milkrecordings": [15.0, 25.0, 30.0]})
    assert r.status_code == 422


def test_characteristic_missing_milkrecordings(api: httpx.Client):
    """Missing required field milkrecordings should return 422."""
    r = api.post("/characteristic", json={"dim": [10, 30, 60]})
    assert r.status_code == 422


def test_characteristic_empty_body(api: httpx.Client):
    """Empty request body should return 422."""
    r = api.post("/characteristic", json={})
    assert r.status_code == 422


def test_characteristic_invalid_type(api: httpx.Client, sample_data: dict):
    """Invalid characteristic name should return 422."""
    r = api.post(
        "/characteristic",
        json={
            **sample_data,
            "characteristic": "nonexistent",
        },
    )
    assert r.status_code == 422


def test_characteristic_invalid_model(api: httpx.Client, sample_data: dict):
    """Invalid model name should return 422."""
    r = api.post(
        "/characteristic",
        json={
            **sample_data,
            "model": "nonexistent",
        },
    )
    assert r.status_code == 422


def test_characteristic_mismatched_lengths(api: httpx.Client):
    """dim and milkrecordings with different lengths should return 422."""
    r = api.post(
        "/characteristic",
        json={
            "dim": [10, 30, 60, 90],
            "milkrecordings": [15.0, 25.0],
        },
    )
    assert r.status_code == 422


def test_characteristic_empty_lists(api: httpx.Client):
    """Empty dim and milkrecordings should return 422."""
    r = api.post("/characteristic", json={"dim": [], "milkrecordings": []})
    assert r.status_code == 422


def test_characteristic_parity_zero(api: httpx.Client, sample_data: dict):
    """Parity 0 violates ge=1 constraint -- should return 422."""
    r = api.post("/characteristic", json={**sample_data, "parity": 0})
    assert r.status_code == 422


def test_characteristic_lactation_length_zero(api: httpx.Client, sample_data: dict):
    """Lactation length 0 violates ge=1 constraint -- should return 422."""
    r = api.post(
        "/characteristic",
        json={
            **sample_data,
            "lactation_length": 0,
        },
    )
    assert r.status_code == 422


def test_characteristic_wrong_method(api: httpx.Client):
    """GET on a POST-only endpoint should return 405."""
    r = api.get("/characteristic")
    assert r.status_code == 405
