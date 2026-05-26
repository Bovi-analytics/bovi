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


def test_characteristic_allows_null_value(api: httpx.Client, sample_data: dict, monkeypatch):
    """A non-sensible characteristic result should be returned as JSON null."""
    import main

    monkeypatch.setattr(main, "calculate_characteristic", lambda **kwargs: None)

    r = api.post("/characteristic", json=sample_data)

    assert r.status_code == 200
    assert r.json() == {"value": None}


def test_characteristic_batch_returns_ordered_values(
    api: httpx.Client, sample_data: dict, monkeypatch: pytest.MonkeyPatch
):
    import main

    calls = []

    def fake_calculate_characteristic(**kwargs):
        calls.append(kwargs)
        return len(calls) * 10.0

    monkeypatch.setattr(main, "calculate_characteristic", fake_calculate_characteristic)

    r = api.post(
        "/characteristic/batch",
        json={
            "items": [
                {**sample_data, "id": "peak", "characteristic": "peak_yield"},
                {**sample_data, "id": "total", "characteristic": "cumulative_milk_yield"},
            ]
        },
    )

    assert r.status_code == 200
    assert r.json() == {
        "results": [
            {"id": "peak", "value": 10.0},
            {"id": "total", "value": 20.0},
        ]
    }
    assert [call["characteristic"] for call in calls] == [
        "peak_yield",
        "cumulative_milk_yield",
    ]


def test_characteristic_batch_bayesian_milkbot_uses_configured_key(
    api: httpx.Client, sample_data: dict, monkeypatch: pytest.MonkeyPatch
):
    import main

    calls = []

    def fake_calculate_characteristic(**kwargs):
        calls.append(kwargs)
        return 123.0

    monkeypatch.setattr(main.settings, "milkbot_key", "test-key")
    monkeypatch.setattr(main, "calculate_characteristic", fake_calculate_characteristic)

    r = api.post(
        "/characteristic/batch",
        json={
            "items": [
                {
                    **sample_data,
                    "id": "cow-1",
                    "model": "milkbot",
                    "characteristic": "cumulative_milk_yield",
                    "fitting": "bayesian",
                    "breed": "J",
                    "parity": 2,
                    "continent": "CHEN",
                }
            ]
        },
    )

    assert r.status_code == 200
    assert r.json() == {"results": [{"id": "cow-1", "value": 123.0}]}
    assert calls[0]["key"] == "test-key"
    assert calls[0]["fitting"] == "bayesian"
    assert calls[0]["breed"] == "J"
    assert calls[0]["parity"] == 2
    assert calls[0]["continent"] == "USA"
    assert calls[0]["custom_priors"] == "CHEN"


def test_characteristic_batch_rejects_empty_items(api: httpx.Client):
    r = api.post("/characteristic/batch", json={"items": []})
    assert r.status_code == 422


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


def test_characteristic_bayesian_milkbot_uses_configured_key(
    api: httpx.Client, sample_data: dict, monkeypatch: pytest.MonkeyPatch
):
    import main

    calls = []

    def fake_calculate_characteristic(**kwargs):
        calls.append(kwargs)
        return 123.0

    monkeypatch.setattr(main.settings, "milkbot_key", "test-key")
    monkeypatch.setattr(main, "calculate_characteristic", fake_calculate_characteristic)

    r = api.post(
        "/characteristic",
        json={
            **sample_data,
            "model": "milkbot",
            "characteristic": "cumulative_milk_yield",
            "fitting": "bayesian",
            "breed": "J",
            "parity": 2,
            "continent": "CHEN",
        },
    )

    assert r.status_code == 200
    assert r.json()["value"] == 123.0
    assert calls[0]["key"] == "test-key"
    assert calls[0]["fitting"] == "bayesian"
    assert calls[0]["breed"] == "J"
    assert calls[0]["parity"] == 2
    assert calls[0]["continent"] == "USA"
    assert calls[0]["custom_priors"] == "CHEN"


def test_characteristic_wrong_method(api: httpx.Client):
    """GET on a POST-only endpoint should return 405."""
    r = api.get("/characteristic")
    assert r.status_code == 405
