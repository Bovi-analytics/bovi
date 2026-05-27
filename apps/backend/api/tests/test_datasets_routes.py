"""Tests for the preset datasets routes, in particular the herd-stats endpoint."""

from __future__ import annotations

import json
import math

import pytest
from azure.core.exceptions import AzureError
from bovi_api.herd_stats_ingestion import CowRecord, aggregate_test_day_records
from bovi_api.routes import datasets as datasets_route
from bovi_api.routes.datasets import PresetCow, PresetDatasetResponse
from bovi_api.settings import Settings
from fastapi import HTTPException


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


def _write_local_preset(tmp_path, monkeypatch, dataset="aurora", size="small", period="mixed"):
    presets_dir = tmp_path / "presets"
    preset_dir = presets_dir / dataset
    preset_dir.mkdir(parents=True)
    preset_path = preset_dir / ("full.json" if dataset == "icar" else f"{size}_{period}.json")
    payload = {
        "dataset": dataset,
        "size": size,
        "period": period,
        "cow_count": 1,
        "cows": [
            {
                "cow_id": "874_2",
                "display_name": "Cow 874 - parity 2",
                "parity": 2,
                "dim": [35, 65],
                "milk_kg": [51.26, 45.36],
            }
        ],
    }
    if dataset == "icar":
        payload["actual_yields"] = {"874_2": 10573.1}
    preset_path.write_text(json.dumps(payload))
    monkeypatch.setattr(datasets_route, "_LOCAL_PRESETS_DIR", presets_dir)
    return payload


def test_preset_blob_path_uses_canonical_data_prefix():
    assert (
        datasets_route._preset_blob_path("aurora", "small", "recent")
        == "data/datasets/presets/aurora/small_recent.json"
    )
    assert (
        datasets_route._preset_blob_path("icar", "full", "all")
        == "data/datasets/presets/icar/full.json"
    )


def test_fetch_preset_cows_falls_back_to_local_preset(tmp_path, monkeypatch):
    _write_local_preset(tmp_path, monkeypatch)
    preset = datasets_route.fetch_preset_cows(
        "aurora",
        "small",
        "mixed",
        Settings(
            connection_string="",
            storage_account_name_icar="",
            storage_account_key_icar="",
            storage_account_container_icar="",
        ),
    )

    assert preset.dataset == "aurora"
    assert preset.size == "small"
    assert preset.period == "mixed"
    assert preset.cow_count == 1
    assert preset.cows[0].cow_id == "874_2"
    assert preset.cows[0].milk_kg[0] == pytest.approx(51.26)


def test_fetch_preset_cows_uses_local_preset_when_blob_is_unreachable(tmp_path, monkeypatch):
    _write_local_preset(tmp_path, monkeypatch, dataset="sunnyside", size="small", period="mixed")

    def _raise_blob_error(*args, **kwargs):
        raise AzureError("network unavailable")

    monkeypatch.setattr(datasets_route, "_fetch_blob_preset", _raise_blob_error)

    preset = datasets_route.fetch_preset_cows(
        "sunnyside",
        "small",
        "mixed",
        Settings(connection_string="UseDevelopmentStorage=true"),
    )

    assert preset.dataset == "sunnyside"
    assert preset.cow_count == 1
    assert preset.cows[0].cow_id == "874_2"


def test_fetch_preset_cows_returns_503_when_blob_and_local_preset_are_unavailable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(datasets_route, "_LOCAL_PRESETS_DIR", tmp_path / "missing")

    with pytest.raises(HTTPException) as exc_info:
        datasets_route.fetch_preset_cows(
            "aurora",
            "small",
            "mixed",
            Settings(
                connection_string="",
                storage_account_name_icar="",
                storage_account_key_icar="",
                storage_account_container_icar="",
            ),
        )

    assert exc_info.value.status_code == 503
    assert "local fallback failed" in exc_info.value.detail
