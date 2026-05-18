"""Tests for ISLC and best-predict yield endpoints."""

import httpx


def test_islc_single_lactation(api: httpx.Client, sample_data: dict):
    r = api.post("/islc", json=sample_data)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert result["test_id"] == 0
    assert result["total_305_yield"] > 0


def test_best_predict_single_lactation(api: httpx.Client, sample_data: dict):
    r = api.post("/best-predict", json=sample_data)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert result["test_id"] == 0
    assert result["total_305_yield"] > 0


def test_best_predict_multiple_lactations(api: httpx.Client):
    r = api.post(
        "/best-predict",
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
    assert {row["test_id"] for row in data["results"]} == {1, 2}
    assert all(row["total_305_yield"] > 0 for row in data["results"])


def test_islc_mismatched_lengths(api: httpx.Client):
    r = api.post(
        "/islc",
        json={
            "dim": [10, 30, 60],
            "milkrecordings": [15.0, 25.0],
        },
    )
    assert r.status_code == 422


def test_best_predict_wrong_method(api: httpx.Client):
    r = api.get("/best-predict")
    assert r.status_code == 405
