"""Tests for the /test-interval endpoint (ICAR Test Interval Method)."""

import httpx
import pytest


def test_test_interval_single_lactation(api: httpx.Client, sample_data: dict):
    r = api.post("/test-interval", json=sample_data)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert result["test_id"] == 1
    assert result["total_305_yield"] > 0


def test_test_interval_multiple_lactations(api: httpx.Client):
    r = api.post(
        "/test-interval",
        json={
            "dim": [10, 30, 60, 90, 120, 10, 30, 60, 90, 120],
            "milkrecordings": [
                15.0,
                25.0,
                30.0,
                28.0,
                26.0,
                20.0,
                30.0,
                35.0,
                32.0,
                28.0,
            ],
            "test_ids": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 2
    ids = {r["test_id"] for r in data["results"]}
    assert ids == {1, 2}
    assert all(r["total_305_yield"] > 0 for r in data["results"])


def test_test_interval_defaults(api: httpx.Client, sample_data: dict):
    """Only required fields -- defaults should work."""
    r = api.post("/test-interval", json=sample_data)
    assert r.status_code == 200
    assert isinstance(r.json()["results"][0]["total_305_yield"], float)


def test_test_interval_missing_dim(api: httpx.Client):
    """Missing required field dim should return 422."""
    r = api.post(
        "/test-interval",
        json={
            "milkrecordings": [15.0, 25.0, 30.0],
        },
    )
    assert r.status_code == 422


def test_test_interval_missing_milkrecordings(api: httpx.Client):
    """Missing required field milkrecordings should return 422."""
    r = api.post(
        "/test-interval",
        json={
            "dim": [10, 30, 60],
        },
    )
    assert r.status_code == 422


def test_test_interval_empty_body(api: httpx.Client):
    """Empty request body should return 422."""
    r = api.post("/test-interval", json={})
    assert r.status_code == 422


@pytest.mark.parametrize(
    "bad_dim",
    [
        "not a list",
        [1, "two", 3],
    ],
)
def test_test_interval_invalid_dim_type(api: httpx.Client, bad_dim):
    """Invalid dim types should return 422."""
    r = api.post(
        "/test-interval",
        json={
            "dim": bad_dim,
            "milkrecordings": [15.0, 25.0, 30.0],
        },
    )
    assert r.status_code == 422


def test_test_interval_mismatched_lengths(api: httpx.Client):
    """dim and milkrecordings with different lengths should return 422."""
    r = api.post(
        "/test-interval",
        json={
            "dim": [10, 30, 60, 90],
            "milkrecordings": [15.0, 25.0],
        },
    )
    assert r.status_code == 422


def test_test_interval_mismatched_test_ids(api: httpx.Client):
    """test_ids with different length than dim should return 422."""
    r = api.post(
        "/test-interval",
        json={
            "dim": [10, 30, 60],
            "milkrecordings": [15.0, 25.0, 30.0],
            "test_ids": [1, 1],
        },
    )
    assert r.status_code == 422


def test_test_interval_empty_lists(api: httpx.Client):
    """Empty dim and milkrecordings should return 422."""
    r = api.post("/test-interval", json={"dim": [], "milkrecordings": []})
    assert r.status_code == 422


def test_test_interval_single_point(api: httpx.Client):
    """Single data point is too few -- should return 422."""
    r = api.post(
        "/test-interval",
        json={
            "dim": [10],
            "milkrecordings": [15.0],
        },
    )
    assert r.status_code == 422


def test_test_interval_wrong_method(api: httpx.Client):
    """GET on a POST-only endpoint should return 405."""
    r = api.get("/test-interval")
    assert r.status_code == 405
