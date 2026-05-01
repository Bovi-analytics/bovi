"""Preset cow-dataset endpoints.

Serves pre-generated JSON blobs from Azure Blob Storage.
Blobs live under preset-datasets/{aurora|sunnyside}/{size}_{period}.json
in the icarwebsite container.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from bovi_api.settings import Settings, get_settings

router = APIRouter(prefix="/datasets", tags=["datasets"])

_CONTAINER = "icarwebsite"
_BLOB_PREFIX = "preset-datasets"

DatasetKey = Literal["aurora", "sunnyside"]
SizeKey = Literal["small", "medium", "large"]
PeriodKey = Literal["recent", "old", "mixed"]

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


def _get_blob_client(settings: Settings) -> BlobServiceClient:
    if not settings.connection_string:
        raise HTTPException(
            status_code=503,
            detail=(
                "Azure Blob Storage is not configured on this server (CONNECTION_STRING missing)."
            ),
        )
    return BlobServiceClient.from_connection_string(settings.connection_string)


def fetch_preset_cows(
    dataset: DatasetKey,
    size: SizeKey,
    period: PeriodKey,
    settings: Settings,
) -> PresetDatasetResponse:
    """Fetch cow data from Azure Blob — shared by routes and benchmark logic.

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
    blob_path = f"{_BLOB_PREFIX}/{dataset}/{size}_{period}.json"
    client = _get_blob_client(settings)
    try:
        data = client.get_blob_client(_CONTAINER, blob_path).download_blob().readall()
    except ResourceNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Preset dataset not found: {blob_path}.",
        )
    payload = json.loads(data)
    return PresetDatasetResponse(**payload)


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
