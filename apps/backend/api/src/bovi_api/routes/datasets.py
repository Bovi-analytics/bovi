"""Preset cow-dataset endpoints.

Serves pre-generated JSON blobs from Azure Blob Storage or local presets.
Azure blobs live under preset-datasets/{aurora|sunnyside}/{size}_{period}.json
in the icarwebsite container. Local presets live under data/datasets/presets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from bovi_api.herd_stats_ingestion import (
    DEFAULT_STAT_RANGES,
    CowRecord,
    aggregate_test_day_records,
    normalize_herd_stats,
)
from bovi_api.settings import Settings, get_settings

router = APIRouter(prefix="/datasets", tags=["datasets"])

_CONTAINER = "icarwebsite"
_BLOB_PREFIX = "preset-datasets"
_LOCAL_PRESETS_DIR = Path(__file__).resolve().parents[6] / "data" / "datasets" / "presets"

DatasetKey = Literal["aurora", "sunnyside", "icar"]
SizeKey = Literal["small", "medium", "large", "full"]
PeriodKey = Literal["recent", "old", "mixed", "all"]

_MANIFEST: dict[str, dict] = {
    "aurora": {
        "label": "Aurora Ridge",
        "description": "AuroraRidge herd, 2023–2025",
        "sizes": {"small": 200, "medium": 1000, "large": 5000},
        "periods": {
            "recent": "2025 records",
            "old": "2023 records",
            "mixed": "2023–2025 mixed",
        },
    },
    "sunnyside": {
        "label": "Sunnyside",
        "description": "Sunnyside herd, 2000–2026",
        "sizes": {"small": 200, "medium": 1000, "large": 5000},
        "periods": {
            "recent": "Records from 2020 onwards",
            "old": "Records before 2010",
            "mixed": "Full period sample",
        },
    },
}


class PresetCow(BaseModel):
    """One cow-lactation record in a preset dataset."""

    cow_id: str
    display_name: str
    parity: int | None
    dim: list[int]
    milk_kg: list[float]


class PresetDatasetResponse(BaseModel):
    """Response for GET /datasets/presets/{dataset}/{size}/{period}."""

    dataset: str
    size: str
    period: str
    cow_count: int
    cows: list[PresetCow]
    actual_yields: dict[str, float] | None = None


class LocalPresetUnavailableError(RuntimeError):
    """Raised when a preset cannot be read from local JSON."""


def _get_blob_client(settings: Settings) -> BlobServiceClient:
    if not settings.connection_string:
        raise HTTPException(
            status_code=503,
            detail=(
                "Azure Blob Storage is not configured on this server (CONNECTION_STRING missing)."
            ),
        )
    return BlobServiceClient.from_connection_string(settings.connection_string)


def _preset_blob_path(dataset: DatasetKey, size: SizeKey, period: PeriodKey) -> str:
    if dataset == "icar":
        return f"{_BLOB_PREFIX}/icar/full.json"
    return f"{_BLOB_PREFIX}/{dataset}/{size}_{period}.json"


def _fetch_blob_preset(
    dataset: DatasetKey,
    size: SizeKey,
    period: PeriodKey,
    settings: Settings,
) -> PresetDatasetResponse:
    blob_path = _preset_blob_path(dataset, size, period)
    client = _get_blob_client(settings)
    data = client.get_blob_client(_CONTAINER, blob_path).download_blob().readall()
    return PresetDatasetResponse(**json.loads(data))


def _fetch_local_preset(
    dataset: DatasetKey,
    size: SizeKey,
    period: PeriodKey,
) -> PresetDatasetResponse:
    local_path = _LOCAL_PRESETS_DIR / _preset_blob_path(dataset, size, period).removeprefix(
        f"{_BLOB_PREFIX}/"
    )
    if not local_path.exists():
        raise LocalPresetUnavailableError(f"Local preset not found: {local_path}")
    return PresetDatasetResponse(**json.loads(local_path.read_text()))


def fetch_preset_cows(
    dataset: DatasetKey,
    size: SizeKey,
    period: PeriodKey,
    settings: Settings,
) -> PresetDatasetResponse:
    """Fetch cow data from Azure Blob - shared by routes and benchmark logic.

    Args:
        dataset (DatasetKey): Preset dataset identifier (e.g. "aurora", "sunnyside").
        size (SizeKey): Dataset size key ("small", "medium", "large").
        period (PeriodKey): Period key ("recent", "old", "mixed").
        settings (Settings): Application settings providing the Azure connection string.

    Returns:
        PresetDatasetResponse: Parsed preset dataset payload.

    Raises:
        HTTPException: 404 if the preset blob does not exist; 503 if blob storage is not configured.

    """
    blob_error: HTTPException | AzureError | None = None
    if settings.connection_string:
        try:
            return _fetch_blob_preset(dataset, size, period, settings)
        except ResourceNotFoundError as exc:
            blob_error = exc
        except AzureError as exc:
            blob_error = exc
    else:
        blob_error = HTTPException(
            status_code=503,
            detail=(
                "Azure Blob Storage is not configured on this server (CONNECTION_STRING missing)."
            ),
        )

    try:
        return _fetch_local_preset(dataset, size, period)
    except LocalPresetUnavailableError as exc:
        if isinstance(blob_error, ResourceNotFoundError):
            blob_path = _preset_blob_path(dataset, size, period)
            raise HTTPException(
                status_code=404,
                detail=f"Preset dataset not found: {blob_path}.",
            ) from exc
        raise HTTPException(
            status_code=503,
            detail=f"Preset dataset service unavailable and local fallback failed: {exc}",
        ) from exc


@router.get("/presets")
def list_presets() -> dict:
    """Return a static manifest of all available preset dataset combinations."""
    return _MANIFEST


@router.get("/presets/{dataset}/{size}/{period}", response_model=PresetDatasetResponse)
def get_preset(
    dataset: DatasetKey,
    size: SizeKey,
    period: PeriodKey,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PresetDatasetResponse:
    """Fetch a pre-generated cow-dataset JSON blob from Azure Blob Storage."""
    return fetch_preset_cows(dataset, size, period, settings)


class PresetHerdStatsResponse(BaseModel):
    """Aggregated herd stats computed from a preset dataset slice."""

    dataset: str
    size: str
    period: str
    parity: int | None
    cow_count: int
    raw_stats: dict[str, float]
    stats: dict[str, float]
    warnings: list[str]


@router.get(
    "/presets/{dataset}/{size}/{period}/herd-stats",
    response_model=PresetHerdStatsResponse,
)
def get_preset_herd_stats(
    dataset: DatasetKey,
    size: SizeKey,
    period: PeriodKey,
    settings: Annotated[Settings, Depends(get_settings)],
    parity: Annotated[int | None, Query(ge=1, le=12)] = None,
) -> PresetHerdStatsResponse:
    """Compute the canonical 10 herd stats from a preset dataset.

    Slices the preset dataset (optionally by parity), aggregates per-cow
    test-day records into herd-level means using the same logic as the CSV
    ingestion path, and returns both the raw values and normalized [0, 1]
    values suitable for the autoencoder ``herd_stats`` field.
    """
    preset = fetch_preset_cows(dataset, size, period, settings)

    cows = [
        CowRecord(cow_id=c.cow_id, parity=c.parity, dim=c.dim, milk_kg=c.milk_kg)
        for c in preset.cows
        if parity is None or c.parity == parity
    ]
    if not cows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No cows in {dataset}/{size}/{period}"
                + (f" with parity={parity}" if parity is not None else "")
                + "."
            ),
        )

    raw_stats, warnings = aggregate_test_day_records(cows)
    normalized = normalize_herd_stats(raw_stats, DEFAULT_STAT_RANGES)

    return PresetHerdStatsResponse(
        dataset=dataset,
        size=size,
        period=period,
        parity=parity,
        cow_count=len(cows),
        raw_stats=raw_stats,
        stats=normalized,
        warnings=warnings,
    )
