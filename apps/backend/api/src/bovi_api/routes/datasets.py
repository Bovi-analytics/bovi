"""Preset cow-dataset endpoints.

Serves pre-generated JSON blobs from Azure Blob Storage or local presets.
Azure blobs live under data/datasets/presets/{aurora|sunnyside}/{size}_{period}.json.
Local presets live under data/datasets/presets.
"""

from __future__ import annotations

import json
import logging
import re
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
logger = logging.getLogger(__name__)

_BLOB_PREFIX = "data/datasets/presets"
_LOCAL_PRESETS_DIR = Path(__file__).resolve().parents[6] / "data" / "datasets" / "presets"

DatasetKey = Literal["aurora", "sunnyside", "icar"]
SizeKey = Literal["small", "medium", "large", "full"]
PeriodKey = Literal["recent", "old", "mixed", "all"]
_PUBLIC_DATASETS: tuple[DatasetKey, ...] = ("aurora", "sunnyside")
_PUBLIC_SIZES: tuple[SizeKey, ...] = ("small", "medium", "large")
_PUBLIC_PERIODS: tuple[PeriodKey, ...] = ("recent", "old", "mixed")
_COW_COUNT_RE = re.compile(rb'"cow_count"\s*:\s*(\d+)')

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
    herd_id: int | None = None
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


class PresetCountsResponse(BaseModel):
    """Cow counts for every public size/period combination of one preset dataset."""

    dataset: str
    counts: dict[str, dict[str, int]]


class LocalPresetUnavailableError(RuntimeError):
    """Raised when a preset cannot be read from local JSON."""


def _get_blob_client(settings: Settings) -> BlobServiceClient:
    connection_string = settings.connection_string
    if (
        not connection_string
        and settings.storage_account_name_icar
        and settings.storage_account_key_icar
    ):
        connection_string = (
            "DefaultEndpointsProtocol=https;"
            f"AccountName={settings.storage_account_name_icar};"
            f"AccountKey={settings.storage_account_key_icar};"
            "EndpointSuffix=core.windows.net"
        )
    if not connection_string:
        raise HTTPException(
            status_code=503,
            detail=(
                "Azure Blob Storage is not configured on this server "
                "(CONNECTION_STRING or STORAGE_ACCOUNT_NAME_ICAR/STORAGE_ACCOUNT_KEY_ICAR missing)."
            ),
        )
    return BlobServiceClient.from_connection_string(connection_string)


def _get_blob_container(settings: Settings) -> str:
    if not settings.storage_account_container_icar:
        raise HTTPException(
            status_code=503,
            detail=(
                "Azure Blob Storage is not configured on this server "
                "(STORAGE_ACCOUNT_CONTAINER_ICAR missing)."
            ),
        )
    return settings.storage_account_container_icar


def _has_blob_config(settings: Settings) -> bool:
    return bool(
        settings.connection_string
        or (
            settings.storage_account_name_icar
            and settings.storage_account_key_icar
            and settings.storage_account_container_icar
        )
    )


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
    data = (
        client.get_blob_client(_get_blob_container(settings), blob_path).download_blob().readall()
    )
    return PresetDatasetResponse(**json.loads(data))


def _fetch_blob_preset_count(
    dataset: DatasetKey,
    size: SizeKey,
    period: PeriodKey,
    settings: Settings,
) -> int:
    blob_path = _preset_blob_path(dataset, size, period)
    client = _get_blob_client(settings)
    data = (
        client.get_blob_client(_get_blob_container(settings), blob_path)
        .download_blob(offset=0, length=8192)
        .readall()
    )
    match = _COW_COUNT_RE.search(data)
    if not match:
        raise AzureError(f"Preset cow_count not found in blob prefix: {blob_path}")
    return int(match.group(1))


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


def _fetch_local_preset_count(
    dataset: DatasetKey,
    size: SizeKey,
    period: PeriodKey,
) -> int:
    return _fetch_local_preset(dataset, size, period).cow_count


def fetch_preset_counts(dataset: DatasetKey, settings: Settings) -> PresetCountsResponse:
    """Fetch cow counts for public preset size/period combinations."""
    if dataset not in _PUBLIC_DATASETS:
        raise HTTPException(status_code=404, detail=f"Preset counts unavailable for: {dataset}.")

    counts: dict[str, dict[str, int]] = {}
    for period in _PUBLIC_PERIODS:
        counts[period] = {}
        for size in _PUBLIC_SIZES:
            try:
                if _has_blob_config(settings):
                    counts[period][size] = _fetch_blob_preset_count(dataset, size, period, settings)
                else:
                    counts[period][size] = _fetch_local_preset_count(dataset, size, period)
            except ResourceNotFoundError as exc:
                blob_path = _preset_blob_path(dataset, size, period)
                raise HTTPException(
                    status_code=404,
                    detail=f"Preset dataset not found: {blob_path}.",
                ) from exc
            except (AzureError, LocalPresetUnavailableError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Preset counts unavailable for {dataset}: {exc}",
                ) from exc

    return PresetCountsResponse(dataset=dataset, counts=counts)


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
    if _has_blob_config(settings):
        try:
            return _fetch_blob_preset(dataset, size, period, settings)
        except ResourceNotFoundError as exc:
            blob_error = exc
        except AzureError as exc:
            blob_error = exc
        logger.warning(
            "Preset blob fetch failed; trying local fallback",
            extra={
                "dataset": dataset,
                "size": size,
                "period": period,
                "blob_path": _preset_blob_path(dataset, size, period),
                "error": str(blob_error),
            },
        )
    else:
        blob_error = HTTPException(
            status_code=503,
            detail=(
                "Azure Blob Storage is not configured on this server "
                "(CONNECTION_STRING or ICAR storage settings missing)."
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


@router.get("/presets/{dataset}/counts", response_model=PresetCountsResponse)
def get_preset_counts(
    dataset: DatasetKey,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PresetCountsResponse:
    """Return cow counts for all size/period combinations of a preset dataset."""
    return fetch_preset_counts(dataset, settings)


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
