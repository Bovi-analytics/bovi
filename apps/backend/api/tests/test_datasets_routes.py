"""Tests for the preset datasets routes, in particular the herd-stats endpoint."""

from __future__ import annotations

import math

import pytest
from bovi_api.herd_stats_ingestion import CowRecord, aggregate_test_day_records
from bovi_api.routes import datasets as datasets_route
from bovi_api.routes.datasets import PresetCow, PresetDatasetResponse


def _fake_preset(cows: list[PresetCow]) -> PresetDatasetResponse:
    return PresetDatasetResponse(
        dataset="sunnyside",
        size="small",
        period="recent",
        cow_count=len(cows),
        cows=cows,
    )


@pytest.fixture()
def fake_cows() -> list[PresetCow]:
    """Two parity-2 cows + one parity-3 cow with simple test-day curves."""
    dims = list(range(10, 310, 30))  # 10, 40, ..., 280  (10 points)
    return [
        PresetCow(
            cow_id="A1",
            display_name="A1",
            parity=2,
            dim=dims,
            milk_kg=[30.0, 38.0, 35.0, 33.0, 30.0, 27.0, 24.0, 21.0, 18.0, 15.0],
        ),
        PresetCow(
            cow_id="A2",
            display_name="A2",
            parity=2,
            dim=dims,
            milk_kg=[28.0, 36.0, 34.0, 32.0, 29.0, 26.0, 23.0, 20.0, 17.0, 14.0],
        ),
        PresetCow(
            cow_id="B1",
            display_name="B1",
            parity=3,
            dim=dims,
            milk_kg=[34.0, 42.0, 39.0, 36.0, 33.0, 30.0, 27.0, 24.0, 21.0, 18.0],
        ),
    ]


def test_preset_herd_stats_matches_aggregator(client, monkeypatch, fake_cows):
    monkeypatch.setattr(
        datasets_route, "fetch_preset_cows", lambda *args, **kwargs: _fake_preset(fake_cows)
    )

    response = client.get("/datasets/presets/sunnyside/small/recent/herd-stats")
    assert response.status_code == 200
    body = response.json()

    assert body["cow_count"] == 3
    assert body["parity"] is None

    expected_raw, _ = aggregate_test_day_records(
        [CowRecord(c.cow_id, c.parity, c.dim, c.milk_kg) for c in fake_cows]
    )
    for key, value in expected_raw.items():
        assert body["raw_stats"][key] == pytest.approx(value)

    for key, value in body["stats"].items():
        assert 0.0 <= value <= 1.0
        assert not math.isnan(value)


def test_preset_herd_stats_filters_by_parity(client, monkeypatch, fake_cows):
    monkeypatch.setattr(
        datasets_route, "fetch_preset_cows", lambda *args, **kwargs: _fake_preset(fake_cows)
    )

    response = client.get("/datasets/presets/sunnyside/small/recent/herd-stats?parity=2")
    assert response.status_code == 200
    body = response.json()

    assert body["cow_count"] == 2
    assert body["parity"] == 2

    expected_raw, _ = aggregate_test_day_records(
        [CowRecord(c.cow_id, c.parity, c.dim, c.milk_kg) for c in fake_cows if c.parity == 2]
    )
    assert body["raw_stats"]["AchievedMilk"] == pytest.approx(expected_raw["AchievedMilk"])


def test_preset_herd_stats_no_matching_cows_returns_404(client, monkeypatch, fake_cows):
    monkeypatch.setattr(
        datasets_route, "fetch_preset_cows", lambda *args, **kwargs: _fake_preset(fake_cows)
    )
    response = client.get("/datasets/presets/sunnyside/small/recent/herd-stats?parity=9")
    assert response.status_code == 404
