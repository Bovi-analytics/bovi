"""Preset cow-dataset endpoints.

Serves pre-generated JSON blobs from Azure Blob Storage.
Blobs live under preset-datasets/{aurora|sunnyside}/{size}_{period}.json
in the icarwebsite container.
"""

from __future__ import annotations

import json
import random
import zipfile
from io import StringIO
from pathlib import Path
from typing import Annotated, Literal, cast
from xml.etree import ElementTree

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
_LOCAL_RAW_DIR = Path(__file__).resolve().parents[6] / "data" / "raw"
_LBS_TO_KG = 0.45359237
_SIZES: dict[str, int] = {"small": 200, "medium": 1000, "large": 5000}

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

_LOCAL_DATASET_CONFIGS: dict[str, dict] = {
    "aurora": {
        "filenames": [
            "AuroraTDM23_26.csv",
            "AuroraTDM23_26_prepped.csv",
            "MilkRecordingsAuroraRidgeDairy.CSV",
        ],
        "id_col": "ID",
        "parity_col": "LACT",
        "dim_col": "DIM",
        "milk_col": "MILK",
        "date_col": "TestDate",
        "bdat_col": "BDAT",
        "periods": {
            "recent": ("2025-01-01", None),
            "old": (None, "2023-12-31"),
            "mixed": (None, None),
        },
    },
    "sunnyside": {
        "filenames": [
            "MilkRecordingsSunnyside.csv",
            "MilkRecordingsSunnySide.CSV",
        ],
        "id_col": "ID",
        "parity_col": "LACT",
        "dim_col": "DIM",
        "milk_col": "MILK",
        "date_col": "TestDate",
        "bdat_col": None,
        "periods": {
            "recent": ("2020-01-01", None),
            "old": (None, "2009-12-31"),
            "mixed": (None, None),
        },
    },
}

_LOCAL_ICAR_FILES = {
    "test": ["TestDataSet.csv", "TestDataSet(in).csv"],
    "actual_yields": ["ActualMilkYields.csv", "ActualMilkYieldsICAR platform.xlsx"],
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
    """Raised when a preset cannot be built from local raw data."""


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


def _read_local_table(path: Path):
    import pandas as pd

    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path)
        except ImportError:
            df = _read_semicolon_rows_from_xlsx(path)
        if len(df.columns) == 1 and df.iloc[:, 0].astype(str).str.contains(";").any():
            rows = "\n".join(df.iloc[:, 0].dropna().astype(str).tolist())
            df = pd.read_csv(StringIO(rows), sep=";")
    else:
        sample = path.read_bytes()[:4096]
        first_line = sample.splitlines()[0].decode("utf-8", errors="replace")
        sep = ";" if first_line.count(";") > first_line.count(",") else ","
        df = pd.read_csv(path, sep=sep, quotechar='"', low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_semicolon_rows_from_xlsx(path: Path):
    import pandas as pd

    with zipfile.ZipFile(path) as archive:
        shared_strings = archive.read("xl/sharedStrings.xml")
    root = ElementTree.fromstring(shared_strings)
    rows = [
        "".join(node.itertext()).strip()
        for node in root.iter()
        if node.tag.endswith("}si") or node.tag == "si"
    ]
    semicolon_rows = [row for row in rows if ";" in row]
    if not semicolon_rows:
        raise LocalPresetUnavailableError(
            f"{path.name} requires openpyxl or semicolon-delimited shared strings."
        )
    return pd.read_csv(StringIO("\n".join(semicolon_rows)), sep=";")


def _find_local_file(candidates: list[str]) -> Path | None:
    for filename in candidates:
        path = _LOCAL_RAW_DIR / filename
        if path.exists():
            return path
    return None


def _stratified_sample(lactations: list[dict], n: int, seed: str) -> list[dict]:
    if len(lactations) <= n:
        return lactations

    by_parity: dict[int | None, list[dict]] = {}
    for lactation in lactations:
        by_parity.setdefault(lactation["parity"], []).append(lactation)

    rng = random.Random(seed)
    selected: list[dict] = []
    total = len(lactations)
    for group in by_parity.values():
        quota = max(1, round(n * len(group) / total))
        selected.extend(rng.sample(group, min(quota, len(group))))

    rng.shuffle(selected)
    if len(selected) > n:
        return selected[:n]

    remainder = [lactation for lactation in lactations if lactation not in selected]
    selected.extend(rng.sample(remainder, min(n - len(selected), len(remainder))))
    return selected


def _build_local_lactations(config: dict, period: str) -> list[dict]:
    import pandas as pd

    path = _find_local_file(config["filenames"])
    if path is None:
        raise LocalPresetUnavailableError(
            f"Local raw data missing in {_LOCAL_RAW_DIR}: {', '.join(config['filenames'])}"
        )

    df = _read_local_table(path)
    id_col = config["id_col"]
    parity_col = config["parity_col"]
    dim_col = config["dim_col"]
    milk_col = config["milk_col"]
    date_col = config["date_col"]
    bdat_col = config["bdat_col"]
    required = [id_col, parity_col, dim_col, milk_col, date_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise LocalPresetUnavailableError(f"{path.name} missing columns: {', '.join(missing)}")

    df[dim_col] = pd.to_numeric(df[dim_col], errors="coerce")
    milk_series = df[milk_col].astype(str).str.strip().str.replace(",", ".", regex=False)
    df["_milk_kg"] = cast(pd.Series, pd.to_numeric(milk_series, errors="coerce")) * _LBS_TO_KG
    df[parity_col] = pd.to_numeric(df[parity_col], errors="coerce")
    df = df.dropna(subset=[id_col, dim_col, "_milk_kg"])
    df = cast(pd.DataFrame, df[df[dim_col] >= 0])
    df = cast(pd.DataFrame, df[df["_milk_kg"] > 0])
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False, format="mixed")
    df["_bdat"] = (
        pd.to_datetime(df[bdat_col], errors="coerce", dayfirst=False, format="mixed")
        if bdat_col and bdat_col in df.columns
        else pd.NaT
    )
    df["_id"] = df[id_col].astype(str).str.strip()
    df["_parity"] = df[parity_col].astype("Int64")

    low, high = config["periods"][period]
    lactations: list[dict] = []
    for group_key, group in df.groupby(["_id", "_parity"], sort=False):
        cow_id, parity_val = cast(tuple[str, object], group_key)
        group_sorted = group.sort_values(dim_col)
        latest_date = group_sorted["_date"].max()
        if low and not bool(pd.isna(latest_date)) and latest_date < pd.Timestamp(low):
            continue
        if high and not bool(pd.isna(latest_date)) and latest_date > pd.Timestamp(high):
            continue

        parity_int = int(cast(int, parity_val)) if not bool(pd.isna(parity_val)) else None
        bdat_year: int | None = None
        if bdat_col:
            bdat_values = group_sorted["_bdat"].dropna()
            if not bdat_values.empty:
                bdat_year = int(bdat_values.iloc[0].year)
        display_name = (
            f"Cow {cow_id} (b. {bdat_year}) - parity {parity_int}"
            if bdat_year is not None
            else f"Cow {cow_id} - parity {parity_int}"
        )

        lactations.append(
            {
                "cow_id": f"{cow_id}_{parity_int}",
                "display_name": display_name,
                "parity": parity_int,
                "dim": group_sorted[dim_col].astype(int).tolist(),
                "milk_kg": [round(float(v), 2) for v in group_sorted["_milk_kg"].tolist()],
            }
        )
    return lactations


def _norm_id(value: object) -> str:
    try:
        numeric = float(value)  # type: ignore[arg-type]
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _build_local_icar_preset() -> PresetDatasetResponse:
    import pandas as pd

    test_path = _find_local_file(_LOCAL_ICAR_FILES["test"])
    yields_path = _find_local_file(_LOCAL_ICAR_FILES["actual_yields"])
    if test_path is None or yields_path is None:
        raise LocalPresetUnavailableError(
            "Local ICAR raw data missing in "
            f"{_LOCAL_RAW_DIR}: {', '.join(_LOCAL_ICAR_FILES['test'])} and "
            f"{', '.join(_LOCAL_ICAR_FILES['actual_yields'])}"
        )

    test_df = _read_local_table(test_path)
    cols = {c.lower(): c for c in test_df.columns}
    id_col = cols.get("testid") or cols.get("cow_id") or cols.get("id")
    parity_col = cols.get("parity") or cols.get("lact")
    dim_col = cols.get("dim") or cols.get("daysinmilk")
    milk_col = cols.get("dailymilkingyield") or cols.get("milk")
    if not all([id_col, parity_col, dim_col, milk_col]):
        raise LocalPresetUnavailableError(
            f"{test_path.name} missing expected columns. Got: {list(test_df.columns)}"
        )

    test_df[dim_col] = pd.to_numeric(test_df[dim_col], errors="coerce")
    test_df[milk_col] = pd.to_numeric(
        test_df[milk_col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    test_df[parity_col] = pd.to_numeric(test_df[parity_col], errors="coerce")
    test_df = test_df.dropna(subset=[id_col, dim_col, milk_col])
    test_df = cast(pd.DataFrame, test_df[test_df[dim_col] >= 0])
    test_df = cast(pd.DataFrame, test_df[test_df[milk_col] > 0])
    test_df["_id"] = test_df[id_col].apply(_norm_id)
    test_df["_parity"] = test_df[parity_col].astype("Int64")

    cows: list[dict] = []
    for group_key, group in test_df.groupby(["_id", "_parity"], sort=False):
        cow_id, parity_val = cast(tuple[str, object], group_key)
        group_sorted = group.sort_values(dim_col)
        parity_int = int(cast(int, parity_val)) if not bool(pd.isna(parity_val)) else None
        cows.append(
            {
                "cow_id": str(cow_id),
                "display_name": f"Cow {cow_id} - parity {parity_int}",
                "parity": parity_int,
                "dim": group_sorted[dim_col].astype(int).tolist(),
                "milk_kg": [round(float(v), 2) for v in group_sorted[milk_col].tolist()],
            }
        )

    yields_df = _read_local_table(yields_path)
    yield_cols = {c.lower(): c for c in yields_df.columns}
    yield_id = yield_cols.get("testid") or yield_cols.get("cow_id") or yield_cols.get("id")
    yield_value = (
        yield_cols.get("totalactualproduction")
        or yield_cols.get("actualproduction")
        or yield_cols.get("yield_305day")
        or yield_cols.get("total_305_yield")
        or yield_cols.get("total actual production")
    )
    if not yield_id or not yield_value:
        raise LocalPresetUnavailableError(
            f"{yields_path.name} missing expected columns. Got: {list(yields_df.columns)}"
        )

    yields_df[yield_value] = pd.to_numeric(
        yields_df[yield_value].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    yields_df = yields_df.dropna(subset=[yield_id, yield_value])
    actual_yields = {
        _norm_id(row[yield_id]): round(float(row[yield_value]), 2)
        for _, row in yields_df.iterrows()
    }
    cows_with_yields = [cow for cow in cows if cow["cow_id"] in actual_yields]
    return PresetDatasetResponse(
        dataset="icar",
        size="full",
        period="all",
        cow_count=len(cows_with_yields),
        cows=[PresetCow(**cow) for cow in cows_with_yields],
        actual_yields={cow["cow_id"]: actual_yields[cow["cow_id"]] for cow in cows_with_yields},
    )


def _fetch_local_preset(
    dataset: DatasetKey,
    size: SizeKey,
    period: PeriodKey,
) -> PresetDatasetResponse:
    generated_path = _LOCAL_RAW_DIR / _preset_blob_path(dataset, size, period)
    if generated_path.exists():
        return PresetDatasetResponse(**json.loads(generated_path.read_text()))

    if dataset == "icar":
        return _build_local_icar_preset()
    if dataset not in _LOCAL_DATASET_CONFIGS:
        raise LocalPresetUnavailableError(f"No local preset generator for dataset: {dataset}")
    if size not in _SIZES:
        raise LocalPresetUnavailableError(f"No local preset generator for size: {size}")
    if period == "all":
        raise LocalPresetUnavailableError(f"No local preset generator for period: {period}")

    lactations = _build_local_lactations(_LOCAL_DATASET_CONFIGS[dataset], period)
    sampled = _stratified_sample(lactations, _SIZES[size], f"{dataset}:{size}:{period}")
    cows = [PresetCow(**cow) for cow in sampled]
    return PresetDatasetResponse(
        dataset=dataset,
        size=size,
        period=period,
        cow_count=len(cows),
        cows=cows,
    )


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
