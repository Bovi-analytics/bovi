"""Benchmark endpoints - ground-truth ALY benchmarking with selectable benchmark + challenger."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from collections import Counter
from typing import Literal, TypedDict
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, model_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from bovi_api.benchmark_ingestion import (
    parse_actual_yields_csv,
    parse_submission_csv,
    parse_test_day_csv,
)
from bovi_api.benchmark_pdf import generate_report_pdf
from bovi_api.benchmark_stats import calculate_comparison_stats_v2
from bovi_api.database import get_session
from bovi_api.models import (
    Challenge,
    ChallengeDetail,
    ChallengeRead,
    Submission,
    SubmissionRead,
)
from bovi_api.routes.datasets import fetch_preset_cows
from bovi_api.routes.proxy import _get_client
from bovi_api.settings import Settings, get_settings
from bovi_api.storage import (
    ArtifactStorage,
    create_bytes_artifact,
    create_json_artifact,
    delete_artifacts_best_effort,
    get_artifact_storage,
    get_optional_artifact_storage,
    load_bytes_artifact,
    load_json_artifact,
)
from bovi_api.upload_limits import ensure_upload_bytes_size, ensure_upload_file_size

logger = logging.getLogger("bovi_api.benchmark")
router = APIRouter(prefix="/benchmark", tags=["benchmark"])

_FAILURE_THRESHOLD = 0.20
_MAX_PLAUSIBLE_LACTATION_YIELD_KG = 100_000.0

CURVE_MODELS = {"wood", "wilmink", "ali_schaeffer", "fischer", "milkbot"}
YIELD_ESTIMATORS = {
    "tim": "/test-interval",
    "islc": "/islc",
    "best_predict": "/best-predict",
}
ALL_MODELS = CURVE_MODELS | set(YIELD_ESTIMATORS) | {"autoencoder"}

_ICAR_DATASET_SOURCES = [
    {
        "role": "test_day_records",
        "label": "Test-day records",
        "filename": "TestDataSet.csv",
    },
    {
        "role": "actual_yields",
        "label": "Ground-truth ALY",
        "filename": "ActualMilkYields.csv",
    },
]

ChallengerOrBenchmark = Literal[
    "wood",
    "wilmink",
    "ali_schaeffer",
    "fischer",
    "milkbot",
    "autoencoder",
    "tim",
    "islc",
    "best_predict",
]


class _ChallengeSummary(TypedDict):
    row_count: int
    cow_count: int
    actual_yield_count: int
    herd_count: int | None
    parity_counts: dict[str, int]


def _csv_data_row_count(content: bytes) -> int:
    text = content.decode("utf-8-sig", errors="ignore")
    return max(0, sum(1 for line in text.splitlines() if line.strip()) - 1)


def _challenge_summary(
    cow_metadata: dict[str, dict],
    actual_yields: dict[str, float] | None,
) -> _ChallengeSummary:
    herd_ids = {
        meta.get("herd_id")
        for meta in cow_metadata.values()
        if meta.get("herd_id") not in (None, "")
    }
    parity_counts = Counter(
        str(meta.get("parity") if meta.get("parity") is not None else "unknown")
        for meta in cow_metadata.values()
    )
    return {
        "row_count": sum(len(meta.get("dim", [])) for meta in cow_metadata.values()),
        "cow_count": len(cow_metadata),
        "actual_yield_count": len(actual_yields or {}),
        "herd_count": len(herd_ids) if herd_ids else None,
        "parity_counts": dict(parity_counts),
    }


async def _load_required_json(
    session: AsyncSession,
    storage: ArtifactStorage | None,
    artifact_id: str | None,
    legacy_value: dict | None,
    label: str,
) -> dict:
    if legacy_value is not None:
        return legacy_value
    if artifact_id is None:
        raise HTTPException(status_code=500, detail=f"{label} is missing.")
    if storage is None:
        raise HTTPException(status_code=503, detail="Blob storage is not configured.")
    payload = await load_json_artifact(session=session, storage=storage, artifact_id=artifact_id)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail=f"{label} artifact is invalid.")
    return payload


def _require_storage(storage: ArtifactStorage | None) -> ArtifactStorage:
    if storage is None:
        raise HTTPException(status_code=503, detail="Blob storage is not configured.")
    return storage


async def _load_challenge_cow_metadata(
    challenge: Challenge,
    session: AsyncSession,
    storage: ArtifactStorage | None,
) -> dict[str, dict]:
    return await _load_required_json(
        session,
        storage,
        challenge.cow_metadata_artifact_id,
        challenge.cow_metadata,
        "Challenge cow metadata",
    )


async def _load_challenge_actual_yields(
    challenge: Challenge,
    session: AsyncSession,
    storage: ArtifactStorage | None,
) -> dict[str, float]:
    return await _load_required_json(
        session,
        storage,
        challenge.actual_yields_artifact_id,
        challenge.actual_yields,
        "Challenge actual yields",
    )


async def _load_submission_yields(
    submission: Submission,
    session: AsyncSession,
    storage: ArtifactStorage | None,
) -> tuple[dict[str, float], dict[str, float]]:
    submitted = await _load_required_json(
        session,
        storage,
        submission.submitted_yields_artifact_id,
        submission.submitted_yields,
        "Submission yields",
    )
    benchmark = await _load_required_json(
        session,
        storage,
        submission.bovi_yields_artifact_id,
        submission.bovi_yields,
        "Benchmark yields",
    )
    return submitted, benchmark


class MilkBotRunOptions(BaseModel):
    """Optional MilkBot fitting options for benchmark model runs."""

    fitting: Literal["frequentist", "bayesian"] = "frequentist"
    breed: Literal["H", "J"] = "H"
    continent: Literal["USA", "EU", "CHEN"] = "USA"


def _actual_yields_are_plausible(actual_yields: dict | None) -> bool:
    if not actual_yields:
        return True
    return all(
        0 < float(value) < _MAX_PLAUSIBLE_LACTATION_YIELD_KG for value in actual_yields.values()
    )


async def _repair_preset_challenge_if_needed(
    challenge: Challenge,
    session: AsyncSession,
    settings: Settings,
) -> bool:
    """Refresh legacy reference-dataset challenges that stored bad concatenated ALY values."""
    if challenge.actual_yields_artifact_id is not None:
        return False
    if challenge.dataset != "icar" or challenge.source != "preset":
        return False
    if _actual_yields_are_plausible(challenge.actual_yields):
        return False

    loop = asyncio.get_running_loop()
    preset = await loop.run_in_executor(None, fetch_preset_cows, "icar", "full", "all", settings)
    if not preset.actual_yields:
        raise HTTPException(
            status_code=500,
            detail="Reference dataset is missing actual_yields - regenerate the preset data.",
        )

    challenge.cow_metadata = {
        cow.cow_id: {
            "parity": cow.parity,
            "herd_id": cow.herd_id,
            "dim": cow.dim,
            "milk_kg": cow.milk_kg,
        }
        for cow in preset.cows
        if cow.cow_id in preset.actual_yields
    }
    challenge.actual_yields = {cid: preset.actual_yields[cid] for cid in challenge.cow_metadata}
    session.add(challenge)
    await session.commit()
    await session.refresh(challenge)
    return True


def _recalculate_submission_stats(
    *,
    submission: Submission,
    cow_metadata: dict[str, dict],
    actual_yields: dict[str, float],
    submitted_yields: dict[str, float],
    benchmark_yields: dict[str, float],
) -> dict:
    parities = {cid: meta.get("parity", 1) or 1 for cid, meta in cow_metadata.items()}
    return calculate_comparison_stats_v2(
        challenger_yields=submitted_yields,
        benchmark_yields=benchmark_yields,
        actual_yields=actual_yields,
        parities=parities,
    )


def _dataset_stats(cow_metadata: dict | None, actual_yields: dict | None) -> dict[str, int | None]:
    """Build the compact dataset summary used by benchmark list/detail views."""
    cow_metadata = cow_metadata or {}
    actual_yields = actual_yields or {}
    herd_ids = {
        meta.get("herd_id")
        for meta in cow_metadata.values()
        if isinstance(meta, dict) and meta.get("herd_id") not in (None, "")
    }
    return {
        "lactation_count": len(cow_metadata),
        "test_day_row_count": sum(
            len(meta.get("dim", [])) for meta in cow_metadata.values() if isinstance(meta, dict)
        ),
        "actual_yield_count": len(actual_yields),
        "herd_count": len(herd_ids) if herd_ids else None,
    }


def _upload_dataset_sources(
    test_day_filename: str | None = None,
    actual_yields_filename: str | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "test_day_records",
            "label": "Test-day records",
            "filename": test_day_filename or "Uploaded test-day CSV",
        },
        {
            "role": "actual_yields",
            "label": "Ground-truth ALY",
            "filename": actual_yields_filename or "Uploaded actual-yields CSV",
        },
    ]


def _fallback_dataset_sources(challenge: Challenge) -> list[dict[str, str]]:
    if challenge.dataset == "icar" and challenge.source == "preset":
        return _ICAR_DATASET_SOURCES
    if challenge.dataset in {"upload", "saved_upload"}:
        return _upload_dataset_sources()
    return [
        {
            "role": "dataset",
            "label": "Dataset",
            "filename": challenge.name or challenge.dataset,
        }
    ]


def _with_dataset_metadata(challenge: Challenge) -> Challenge:
    if not challenge.dataset_sources:
        challenge.dataset_sources = _fallback_dataset_sources(challenge)
    if not challenge.dataset_stats:
        challenge.dataset_stats = _dataset_stats(challenge.cow_metadata, challenge.actual_yields)
    return challenge


# ---------------------------------------------------------------------------
# Model dispatchers
# ---------------------------------------------------------------------------


async def _call_yield_estimator(
    cow_metadata: dict[str, dict],
    settings: Settings,
    path: str,
) -> dict[str, float]:
    """Call a lactation-curves 305-day yield endpoint for all cows."""
    dim_all: list[int] = []
    milk_all: list[float] = []
    test_ids_all: list[str] = []
    for cow_id, meta in cow_metadata.items():
        for d, m in zip(meta["dim"], meta["milk_kg"]):
            dim_all.append(d)
            milk_all.append(m)
            test_ids_all.append(cow_id)

    payload = {"dim": dim_all, "milkrecordings": milk_all, "test_ids": test_ids_all}
    client = _get_client()
    try:
        resp = await client.post(
            f"{settings.lactation_curves_url}{path}",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
    except httpx.RequestError as exc:
        logger.exception("Yield estimator proxy error for %s: %s", path, exc)
        raise HTTPException(
            status_code=502, detail="Upstream lactation-curves service unavailable."
        )

    data = resp.json()
    if isinstance(data, dict) and "results" in data:
        items = data["results"]
    elif isinstance(data, list):
        items = data
    else:
        return {str(k): float(v) for k, v in data.items()}

    out: dict[str, float] = {}
    for item in items:
        cow_id = item.get("test_id") or item.get("cow_id")
        yield_val = item.get("total_305_yield", item.get("yield_305day"))
        if cow_id is None or yield_val is None:
            continue
        out[str(cow_id)] = float(yield_val)
    return out


async def _call_curve_characteristic(
    cow_metadata: dict[str, dict],
    model: str,
    settings: Settings,
    options: MilkBotRunOptions | None = None,
) -> dict[str, float]:
    """Call batch POST /characteristic/batch - failed cows are omitted."""
    client = _get_client()
    items: list[dict] = []
    for cow_id, meta in cow_metadata.items():
        if len(meta.get("dim", [])) < 2:
            continue
        item = {
            "id": cow_id,
            "dim": meta["dim"],
            "milkrecordings": meta["milk_kg"],
            "model": model,
            "characteristic": "cumulative_milk_yield",
            "parity": meta.get("parity") or 1,
            "lactation_length": 305,
        }
        if model == "milkbot" and options is not None:
            item.update(
                {
                    "fitting": options.fitting,
                    "breed": options.breed,
                    "continent": options.continent,
                }
            )
        items.append(item)

    if not items:
        return {}

    try:
        resp = await client.post(
            f"{settings.lactation_curves_url}/characteristic/batch",
            content=json.dumps({"items": items}),
            headers={"Content-Type": "application/json"},
            timeout=300.0,
        )
        resp.raise_for_status()
    except httpx.RequestError as exc:
        logger.exception("Curve characteristic batch proxy error: %s", exc)
        raise HTTPException(
            status_code=502, detail="Upstream lactation-curves service unavailable."
        )
    except httpx.HTTPStatusError:
        return {}

    out: dict[str, float] = {}
    for item in resp.json().get("results", []):
        cow_id = item.get("id")
        val = item.get("value")
        if cow_id is None or val is None:
            continue
        out[str(cow_id)] = float(val)
    return out


async def _call_autoencoder(
    cow_metadata: dict[str, dict],
    settings: Settings,
) -> dict[str, float]:
    """Predict 304-day curves via autoencoder /predict/batch and sum."""
    cow_ids = list(cow_metadata.keys())
    items: list[dict] = []
    for cow_id in cow_ids:
        meta = cow_metadata[cow_id]
        milk: list[float | None] = [None] * 304
        for d, m in zip(meta["dim"], meta["milk_kg"]):
            if 1 <= d <= 304:
                milk[d - 1] = float(m)
        item = {"milk": milk, "parity": meta.get("parity") or 1}
        if meta.get("herd_id") is not None:
            item["herd_id"] = meta["herd_id"]
        items.append(item)

    payload = {"items": items, "imputation_method": "forward_fill"}
    client = _get_client()
    try:
        resp = await client.post(
            f"{settings.lactation_autoencoder_url}/predict/batch",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
    except httpx.RequestError as exc:
        logger.exception("Autoencoder proxy error: %s", exc)
        raise HTTPException(
            status_code=502, detail="Upstream lactation-autoencoder service unavailable."
        )

    data = resp.json()
    results = data.get("results", [])
    out: dict[str, float] = {}
    for cow_id, result in zip(cow_ids, results):
        preds = result.get("predictions") or []
        if not preds:
            continue
        out[cow_id] = float(sum(preds))
    return out


async def _dispatch_model(
    model: str,
    cow_metadata: dict[str, dict],
    settings: Settings,
    options: MilkBotRunOptions | None = None,
) -> dict[str, float]:
    """Run the requested model on all cows and return {cow_id: 305-day yield}."""
    if model in YIELD_ESTIMATORS:
        return await _call_yield_estimator(cow_metadata, settings, YIELD_ESTIMATORS[model])
    if model in CURVE_MODELS:
        return await _call_curve_characteristic(cow_metadata, model, settings, options)
    if model == "autoencoder":
        return await _call_autoencoder(cow_metadata, settings)
    raise HTTPException(status_code=422, detail=f"Unknown model '{model}'.")


# ---------------------------------------------------------------------------
# Challenge endpoints
# ---------------------------------------------------------------------------


class ChallengeCreatePresetBody(BaseModel):
    source: Literal["preset"] = "preset"
    preset: Literal["icar"] = "icar"
    name: str | None = None


class ChallengeCreateSavedDatasetBody(BaseModel):
    name: str
    cow_metadata: dict[str, dict]
    actual_yields: dict[str, float]
    dataset_sources: list[dict] | None = None


@router.post("/challenges", response_model=ChallengeRead, status_code=201)
async def create_challenge_preset(
    body: ChallengeCreatePresetBody,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    storage: ArtifactStorage = Depends(get_artifact_storage),
) -> Challenge:
    """Create a preset-backed challenge with ground-truth ALY."""
    loop = asyncio.get_running_loop()
    preset = await loop.run_in_executor(
        None, fetch_preset_cows, body.preset, "full", "all", settings
    )
    if not preset.actual_yields:
        raise HTTPException(
            status_code=500,
            detail="Reference dataset is missing actual_yields - regenerate the preset blob.",
        )

    cow_metadata: dict[str, dict] = {
        cow.cow_id: {
            "parity": cow.parity,
            "herd_id": cow.herd_id,
            "dim": cow.dim,
            "milk_kg": cow.milk_kg,
        }
        for cow in preset.cows
    }
    # Keep only cows that have ALY (defensive - already filtered by generator)
    cow_metadata = {cid: m for cid, m in cow_metadata.items() if cid in preset.actual_yields}
    actual_yields = {cid: preset.actual_yields[cid] for cid in cow_metadata}
    challenge_uuid = str(uuid4())
    summary = _challenge_summary(cow_metadata, actual_yields)
    prefix = storage.path("challenges", challenge_uuid)
    uploaded = [
        await create_json_artifact(
            session=session,
            storage=storage,
            artifact_kind="challenge_cow_metadata_json",
            entity_type="challenge",
            entity_uuid=challenge_uuid,
            blob_path=f"{prefix}/parsed/cow_metadata.json.gz",
            payload=cow_metadata,
            row_count=summary["row_count"],
            record_count=summary["cow_count"],
        ),
        await create_json_artifact(
            session=session,
            storage=storage,
            artifact_kind="challenge_actual_yields_json",
            entity_type="challenge",
            entity_uuid=challenge_uuid,
            blob_path=f"{prefix}/parsed/actual_yields.json.gz",
            payload=actual_yields,
            record_count=summary["actual_yield_count"],
        ),
    ]
    try:
        await session.flush()
    except Exception:
        await session.rollback()
        await delete_artifacts_best_effort(storage, uploaded)
        raise

    challenge = Challenge(
        uuid=challenge_uuid,
        dataset=body.preset,
        size="full",
        period="all",
        source="preset",
        name=body.name or "Demo dataset",
        cow_metadata=None,
        reference_yields=None,
        actual_yields=None,
        cow_metadata_artifact_id=uploaded[0].id,
        actual_yields_artifact_id=uploaded[1].id,
        row_count=summary["row_count"],
        cow_count=summary["cow_count"],
        actual_yield_count=summary["actual_yield_count"],
        herd_count=summary["herd_count"],
        parity_counts=summary["parity_counts"],
        dataset_sources=_ICAR_DATASET_SOURCES,
        dataset_stats=_dataset_stats(cow_metadata, actual_yields),
    )
    session.add(challenge)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        await delete_artifacts_best_effort(storage, uploaded)
        raise
    await session.refresh(challenge)
    return challenge


@router.post("/challenges/saved-dataset", response_model=ChallengeRead, status_code=201)
async def create_challenge_saved_dataset(
    body: ChallengeCreateSavedDatasetBody,
    session: AsyncSession = Depends(get_session),
    storage: ArtifactStorage = Depends(get_artifact_storage),
) -> Challenge:
    """Create a challenge from a previously parsed upload-backed benchmark dataset."""
    if not body.cow_metadata:
        raise HTTPException(status_code=422, detail="Saved dataset has no test-day records.")
    if not body.actual_yields:
        raise HTTPException(status_code=422, detail="Saved dataset has no actual yields.")

    overlap = sum(1 for cid in body.cow_metadata if cid in body.actual_yields)
    if overlap / len(body.cow_metadata) < 0.80:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Actual yields must cover at least 80% of lactations from the test-day data "
                f"(got {overlap}/{len(body.cow_metadata)})."
            ),
        )
    cow_metadata = {cid: m for cid, m in body.cow_metadata.items() if cid in body.actual_yields}

    actual_yields = {cid: float(body.actual_yields[cid]) for cid in cow_metadata}
    challenge_uuid = str(uuid4())
    summary = _challenge_summary(cow_metadata, actual_yields)
    prefix = storage.path("challenges", challenge_uuid)
    uploaded = [
        await create_json_artifact(
            session=session,
            storage=storage,
            artifact_kind="challenge_cow_metadata_json",
            entity_type="challenge",
            entity_uuid=challenge_uuid,
            blob_path=f"{prefix}/parsed/cow_metadata.json.gz",
            payload=cow_metadata,
            row_count=summary["row_count"],
            record_count=summary["cow_count"],
            metadata_extra={"source": "saved_dataset_payload"},
        ),
        await create_json_artifact(
            session=session,
            storage=storage,
            artifact_kind="challenge_actual_yields_json",
            entity_type="challenge",
            entity_uuid=challenge_uuid,
            blob_path=f"{prefix}/parsed/actual_yields.json.gz",
            payload=actual_yields,
            record_count=summary["actual_yield_count"],
            metadata_extra={"source": "saved_dataset_payload"},
        ),
    ]
    try:
        await session.flush()
    except Exception:
        await session.rollback()
        await delete_artifacts_best_effort(storage, uploaded)
        raise

    challenge = Challenge(
        uuid=challenge_uuid,
        dataset="saved_upload",
        size="custom",
        period="custom",
        source="upload",
        name=body.name,
        cow_metadata=None,
        reference_yields=None,
        actual_yields=None,
        cow_metadata_artifact_id=uploaded[0].id,
        actual_yields_artifact_id=uploaded[1].id,
        row_count=summary["row_count"],
        cow_count=summary["cow_count"],
        actual_yield_count=summary["actual_yield_count"],
        herd_count=summary["herd_count"],
        parity_counts=summary["parity_counts"],
        dataset_sources=body.dataset_sources or _upload_dataset_sources(),
        dataset_stats=_dataset_stats(cow_metadata, actual_yields),
    )
    session.add(challenge)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        await delete_artifacts_best_effort(storage, uploaded)
        raise
    await session.refresh(challenge)
    return challenge


@router.post("/challenges/upload", response_model=ChallengeRead, status_code=201)
async def create_challenge_upload(
    name: str = Form(...),
    test_day_csv: UploadFile = File(...),
    actual_yields_csv: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    storage: ArtifactStorage = Depends(get_artifact_storage),
) -> Challenge:
    """Create an upload-backed challenge: user-supplied test-day records + ground-truth yields."""
    ensure_upload_file_size(test_day_csv, max_size=settings.upload_max_bytes)
    ensure_upload_file_size(actual_yields_csv, max_size=settings.upload_max_bytes)
    test_bytes = await test_day_csv.read()
    aly_bytes = await actual_yields_csv.read()
    ensure_upload_bytes_size(
        test_bytes,
        filename=test_day_csv.filename,
        max_size=settings.upload_max_bytes,
    )
    ensure_upload_bytes_size(
        aly_bytes,
        filename=actual_yields_csv.filename,
        max_size=settings.upload_max_bytes,
    )

    try:
        cow_metadata = parse_test_day_csv(test_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"test-day CSV: {exc}") from exc
    try:
        actual_yields = parse_actual_yields_csv(aly_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"actual-yields CSV: {exc}") from exc

    # Coverage check: ALY must cover at least 80% of lactations in the test-day file
    overlap = sum(1 for cid in cow_metadata if cid in actual_yields)
    if not cow_metadata or overlap / len(cow_metadata) < 0.80:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Actual-yields CSV must cover at least 80% of lactations from the test-day CSV "
                f"(got {overlap}/{len(cow_metadata)})."
            ),
        )
    # Drop cows without ALY
    cow_metadata = {cid: m for cid, m in cow_metadata.items() if cid in actual_yields}
    actual_yields = {cid: actual_yields[cid] for cid in cow_metadata}

    challenge_uuid = str(uuid4())
    summary = _challenge_summary(cow_metadata, actual_yields)
    prefix = storage.path("challenges", challenge_uuid)
    uploaded = [
        await create_bytes_artifact(
            session=session,
            storage=storage,
            artifact_kind="challenge_test_day_csv",
            entity_type="challenge",
            entity_uuid=challenge_uuid,
            blob_path=f"{prefix}/raw/test_day.csv",
            data=test_bytes,
            content_type="text/csv",
            original_filename=test_day_csv.filename,
            row_count=_csv_data_row_count(test_bytes),
        ),
        await create_bytes_artifact(
            session=session,
            storage=storage,
            artifact_kind="challenge_actual_yields_csv",
            entity_type="challenge",
            entity_uuid=challenge_uuid,
            blob_path=f"{prefix}/raw/actual_yields.csv",
            data=aly_bytes,
            content_type="text/csv",
            original_filename=actual_yields_csv.filename,
            row_count=_csv_data_row_count(aly_bytes),
        ),
        await create_json_artifact(
            session=session,
            storage=storage,
            artifact_kind="challenge_cow_metadata_json",
            entity_type="challenge",
            entity_uuid=challenge_uuid,
            blob_path=f"{prefix}/parsed/cow_metadata.json.gz",
            payload=cow_metadata,
            row_count=summary["row_count"],
            record_count=summary["cow_count"],
        ),
        await create_json_artifact(
            session=session,
            storage=storage,
            artifact_kind="challenge_actual_yields_json",
            entity_type="challenge",
            entity_uuid=challenge_uuid,
            blob_path=f"{prefix}/parsed/actual_yields.json.gz",
            payload=actual_yields,
            record_count=summary["actual_yield_count"],
        ),
    ]
    try:
        await session.flush()
    except Exception:
        await session.rollback()
        await delete_artifacts_best_effort(storage, uploaded)
        raise

    challenge = Challenge(
        uuid=challenge_uuid,
        dataset="upload",
        size="custom",
        period="custom",
        source="upload",
        name=name,
        cow_metadata=None,
        reference_yields=None,
        actual_yields=None,
        test_day_file_artifact_id=uploaded[0].id,
        actual_yields_file_artifact_id=uploaded[1].id,
        cow_metadata_artifact_id=uploaded[2].id,
        actual_yields_artifact_id=uploaded[3].id,
        test_day_filename=test_day_csv.filename,
        actual_yields_filename=actual_yields_csv.filename,
        row_count=summary["row_count"],
        cow_count=summary["cow_count"],
        actual_yield_count=summary["actual_yield_count"],
        herd_count=summary["herd_count"],
        parity_counts=summary["parity_counts"],
        dataset_sources=_upload_dataset_sources(test_day_csv.filename, actual_yields_csv.filename),
        dataset_stats=_dataset_stats(cow_metadata, actual_yields),
    )
    session.add(challenge)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        await delete_artifacts_best_effort(storage, uploaded)
        raise
    await session.refresh(challenge)
    return challenge


@router.get("/challenges", response_model=list[ChallengeRead])
async def list_challenges(
    session: AsyncSession = Depends(get_session),
) -> list[Challenge]:
    """List all challenges, newest first."""
    result = await session.execute(
        select(Challenge).order_by(col(Challenge.created_at).desc()).limit(100)
    )
    return [_with_dataset_metadata(challenge) for challenge in result.scalars().all()]


@router.get("/challenges/{challenge_id}", response_model=ChallengeDetail)
async def get_challenge(
    challenge_id: int,
    session: AsyncSession = Depends(get_session),
    storage: ArtifactStorage | None = Depends(get_optional_artifact_storage),
) -> ChallengeDetail:
    """Get a single challenge with full cow metadata."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    cow_metadata = await _load_challenge_cow_metadata(challenge, session, storage)
    actual_yields = await _load_challenge_actual_yields(challenge, session, storage)
    challenge = _with_dataset_metadata(challenge)
    dataset_stats = challenge.dataset_stats or _dataset_stats(cow_metadata, actual_yields)
    return ChallengeDetail(
        id=challenge.id or 0,
        dataset=challenge.dataset,
        size=challenge.size,
        period=challenge.period,
        user_id=challenge.user_id,
        created_at=challenge.created_at,
        name=challenge.name,
        source=challenge.source,
        row_count=challenge.row_count,
        cow_count=challenge.cow_count,
        actual_yield_count=challenge.actual_yield_count,
        ingest_status=challenge.ingest_status,
        dataset_sources=challenge.dataset_sources or _fallback_dataset_sources(challenge),
        dataset_stats=dataset_stats,
        cow_metadata=cow_metadata,
        reference_yields=challenge.reference_yields,
        actual_yields=actual_yields,
    )


@router.get("/challenges/{challenge_id}/export")
async def export_challenge(
    challenge_id: int,
    session: AsyncSession = Depends(get_session),
    storage: ArtifactStorage | None = Depends(get_optional_artifact_storage),
) -> Response:
    """Download a CSV of the challenge's test-day data."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")

    if challenge.test_day_file_artifact_id is not None:
        content = await load_bytes_artifact(
            session=session,
            storage=_require_storage(storage),
            artifact_id=challenge.test_day_file_artifact_id,
        )
        return Response(
            content=content or b"",
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=challenge_{challenge_id}.csv"},
        )

    cow_metadata = await _load_challenge_cow_metadata(challenge, session, storage)
    output = io.StringIO()
    writer = csv.writer(output)
    include_herd_id = any(meta.get("herd_id") not in (None, "") for meta in cow_metadata.values())
    headers = ["TestId", "parity", "dim", "milk_kg"]
    if include_herd_id:
        headers.insert(1, "herd_id")
    writer.writerow(headers)
    for cow_id, meta in cow_metadata.items():
        herd_id = meta.get("herd_id", "")
        parity = meta.get("parity", "")
        for d, m in zip(meta["dim"], meta["milk_kg"]):
            row = [cow_id, parity, d, m]
            if include_herd_id:
                row.insert(1, herd_id)
            writer.writerow(row)

    content = output.getvalue().encode("utf-8")
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=challenge_{challenge_id}.csv"},
    )


# ---------------------------------------------------------------------------
# Submission endpoints
# ---------------------------------------------------------------------------


class SubmissionBoviBody(BaseModel):
    submission_type: Literal["bovi_model"] = "bovi_model"
    challenger: ChallengerOrBenchmark = "wood"
    benchmark: ChallengerOrBenchmark = "tim"
    challenger_options: MilkBotRunOptions | None = None
    benchmark_options: MilkBotRunOptions | None = None
    organization: str | None = None
    country: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _challenger_differs(self) -> "SubmissionBoviBody":
        if self.challenger == self.benchmark:
            raise ValueError("Challenger and benchmark must differ.")
        if self.challenger != "milkbot" and self.challenger_options is not None:
            raise ValueError("challenger_options are only supported for MilkBot.")
        if self.benchmark != "milkbot" and self.benchmark_options is not None:
            raise ValueError("benchmark_options are only supported for MilkBot.")
        return self


@router.post(
    "/challenges/{challenge_id}/submissions", response_model=SubmissionRead, status_code=201
)
async def create_submission_bovi(
    challenge_id: int,
    body: SubmissionBoviBody,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    storage: ArtifactStorage = Depends(get_artifact_storage),
) -> Submission:
    """Run challenger + benchmark models on the cohort and compare both vs ALY."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    cow_metadata = await _load_challenge_cow_metadata(challenge, session, storage)
    actual_yields = await _load_challenge_actual_yields(challenge, session, storage)
    if not actual_yields:
        raise HTTPException(
            status_code=422,
            detail="Challenge has no ground-truth ALY - cannot benchmark.",
        )

    challenger_yields, benchmark_yields = await asyncio.gather(
        _dispatch_model(body.challenger, cow_metadata, settings, body.challenger_options),
        _dispatch_model(body.benchmark, cow_metadata, settings, body.benchmark_options),
    )

    parities = {cid: meta.get("parity", 1) or 1 for cid, meta in cow_metadata.items()}
    failed_cow_ids = [cid for cid in cow_metadata if cid not in challenger_yields]
    stats = calculate_comparison_stats_v2(
        challenger_yields=challenger_yields,
        benchmark_yields=benchmark_yields,
        actual_yields=actual_yields,
        parities=parities,
    )
    submission_uuid = str(uuid4())
    prefix = storage.path("submissions", submission_uuid)
    uploaded = [
        await create_json_artifact(
            session=session,
            storage=storage,
            artifact_kind="submission_submitted_yields_json",
            entity_type="submission",
            entity_uuid=submission_uuid,
            blob_path=f"{prefix}/parsed/submitted_yields.json.gz",
            payload=challenger_yields,
            record_count=len(challenger_yields),
        ),
        await create_json_artifact(
            session=session,
            storage=storage,
            artifact_kind="submission_bovi_yields_json",
            entity_type="submission",
            entity_uuid=submission_uuid,
            blob_path=f"{prefix}/generated/bovi_yields.json.gz",
            payload=benchmark_yields,
            record_count=len(benchmark_yields),
        ),
    ]
    try:
        await session.flush()
    except Exception:
        await session.rollback()
        await delete_artifacts_best_effort(storage, uploaded)
        raise

    submission = Submission(
        uuid=submission_uuid,
        challenge_id=challenge_id,
        submission_type="bovi_model",
        model_type=body.challenger,
        benchmark_model=body.benchmark,
        organization=body.organization,
        country=body.country,
        notes=body.notes,
        run_options={
            "challenger": body.challenger_options.model_dump()
            if body.challenger_options is not None
            else None,
            "benchmark": body.benchmark_options.model_dump()
            if body.benchmark_options is not None
            else None,
        },
        submitted_yields=None,
        bovi_yields=None,
        submitted_yields_artifact_id=uploaded[0].id,
        bovi_yields_artifact_id=uploaded[1].id,
        submitted_yield_count=len(challenger_yields),
        benchmark_yield_count=len(benchmark_yields),
        stats=stats,
        failed_cow_ids=failed_cow_ids,
        failed_count=len(failed_cow_ids),
    )
    session.add(submission)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        await delete_artifacts_best_effort(storage, uploaded)
        raise
    await session.refresh(submission)
    return submission


@router.post(
    "/challenges/{challenge_id}/submissions/upload",
    response_model=SubmissionRead,
    status_code=201,
)
async def create_submission_upload(
    challenge_id: int,
    file: UploadFile = File(...),
    benchmark: str = Form(default="tim"),
    benchmark_fitting: str | None = Form(default=None),
    benchmark_breed: str | None = Form(default=None),
    benchmark_continent: str | None = Form(default=None),
    organization: str | None = Form(default=None),
    country: str | None = Form(default=None),
    calculation_method: str | None = Form(default=None),
    notes: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    storage: ArtifactStorage = Depends(get_artifact_storage),
) -> Submission:
    """Upload a CSV of own-method 305-day yields; benchmark model runs server-side."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    cow_metadata = await _load_challenge_cow_metadata(challenge, session, storage)
    actual_yields = await _load_challenge_actual_yields(challenge, session, storage)
    if not actual_yields:
        raise HTTPException(
            status_code=422,
            detail="Challenge has no ground-truth ALY - cannot benchmark.",
        )
    if benchmark not in ALL_MODELS:
        raise HTTPException(status_code=422, detail=f"Unknown benchmark '{benchmark}'.")
    benchmark_options = None
    has_benchmark_options = any(
        value is not None for value in (benchmark_fitting, benchmark_breed, benchmark_continent)
    )
    if benchmark == "milkbot" and has_benchmark_options:
        try:
            benchmark_options = MilkBotRunOptions.model_validate(
                {
                    "fitting": benchmark_fitting or "frequentist",
                    "breed": benchmark_breed or "H",
                    "continent": benchmark_continent or "USA",
                }
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    elif has_benchmark_options:
        raise HTTPException(
            status_code=422,
            detail="Benchmark options are only supported for MilkBot.",
        )

    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    ensure_upload_file_size(file, max_size=settings.upload_max_bytes)
    content = await file.read()
    ensure_upload_bytes_size(content, filename=filename, max_size=settings.upload_max_bytes)

    try:
        challenger_yields, failed_ids = parse_submission_csv(content, return_failed=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    total = len(challenger_yields) + len(failed_ids)
    if total > 0 and len(failed_ids) / total > _FAILURE_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Too many invalid rows: {len(failed_ids)}/{total} failed "
                f"(>{int(_FAILURE_THRESHOLD * 100)}% threshold)."
            ),
        )

    benchmark_yields = await _dispatch_model(benchmark, cow_metadata, settings, benchmark_options)

    parities = {cid: meta.get("parity", 1) or 1 for cid, meta in cow_metadata.items()}
    stats = calculate_comparison_stats_v2(
        challenger_yields=challenger_yields,
        benchmark_yields=benchmark_yields,
        actual_yields=actual_yields,
        parities=parities,
    )
    submission_uuid = str(uuid4())
    prefix = storage.path("submissions", submission_uuid)
    uploaded = [
        await create_bytes_artifact(
            session=session,
            storage=storage,
            artifact_kind="submission_results_csv",
            entity_type="submission",
            entity_uuid=submission_uuid,
            blob_path=f"{prefix}/raw/results.csv",
            data=content,
            content_type="text/csv",
            original_filename=file.filename,
            row_count=_csv_data_row_count(content),
        ),
        await create_json_artifact(
            session=session,
            storage=storage,
            artifact_kind="submission_submitted_yields_json",
            entity_type="submission",
            entity_uuid=submission_uuid,
            blob_path=f"{prefix}/parsed/submitted_yields.json.gz",
            payload=challenger_yields,
            record_count=len(challenger_yields),
        ),
        await create_json_artifact(
            session=session,
            storage=storage,
            artifact_kind="submission_bovi_yields_json",
            entity_type="submission",
            entity_uuid=submission_uuid,
            blob_path=f"{prefix}/generated/bovi_yields.json.gz",
            payload=benchmark_yields,
            record_count=len(benchmark_yields),
        ),
    ]
    try:
        await session.flush()
    except Exception:
        await session.rollback()
        await delete_artifacts_best_effort(storage, uploaded)
        raise

    submission = Submission(
        uuid=submission_uuid,
        challenge_id=challenge_id,
        submission_type="own_method",
        model_type=None,
        benchmark_model=benchmark,
        organization=organization,
        country=country,
        calculation_method=calculation_method,
        notes=notes,
        run_options={
            "benchmark": benchmark_options.model_dump() if benchmark_options is not None else None
        },
        submitted_yields=None,
        bovi_yields=None,
        input_file_artifact_id=uploaded[0].id,
        submitted_yields_artifact_id=uploaded[1].id,
        bovi_yields_artifact_id=uploaded[2].id,
        input_filename=file.filename,
        row_count=_csv_data_row_count(content),
        submitted_yield_count=len(challenger_yields),
        benchmark_yield_count=len(benchmark_yields),
        stats=stats,
        failed_cow_ids=failed_ids,
        failed_count=len(failed_ids),
    )
    session.add(submission)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        await delete_artifacts_best_effort(storage, uploaded)
        raise
    await session.refresh(submission)
    return submission


@router.get("/submissions", response_model=list[SubmissionRead])
async def list_submissions(
    session: AsyncSession = Depends(get_session),
) -> list[Submission]:
    """List all submissions, newest first."""
    result = await session.execute(
        select(Submission).order_by(col(Submission.created_at).desc()).limit(100)
    )
    return list(result.scalars().all())


@router.get("/submissions/{submission_id}", response_model=SubmissionRead)
async def get_submission(
    submission_id: int,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    storage: ArtifactStorage | None = Depends(get_optional_artifact_storage),
) -> Submission:
    """Get a single submission with stats."""
    sub = await session.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    challenge = await session.get(Challenge, sub.challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    if await _repair_preset_challenge_if_needed(challenge, session, settings):
        cow_metadata = await _load_challenge_cow_metadata(challenge, session, storage)
        actual_yields = await _load_challenge_actual_yields(challenge, session, storage)
        submitted_yields, benchmark_yields = await _load_submission_yields(sub, session, storage)
        sub.stats = _recalculate_submission_stats(
            submission=sub,
            cow_metadata=cow_metadata,
            actual_yields=actual_yields,
            submitted_yields=submitted_yields,
            benchmark_yields=benchmark_yields,
        )
        session.add(sub)
        await session.commit()
        await session.refresh(sub)
    return sub


@router.get("/submissions/{submission_id}/report")
async def download_report(
    submission_id: int,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    storage: ArtifactStorage | None = Depends(get_optional_artifact_storage),
) -> Response:
    """Download a PDF report for a submission."""
    sub = await session.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    challenge = await session.get(Challenge, sub.challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    if await _repair_preset_challenge_if_needed(challenge, session, settings):
        cow_metadata = await _load_challenge_cow_metadata(challenge, session, storage)
        actual_yields = await _load_challenge_actual_yields(challenge, session, storage)
        submitted_yields, benchmark_yields = await _load_submission_yields(sub, session, storage)
        sub.stats = _recalculate_submission_stats(
            submission=sub,
            cow_metadata=cow_metadata,
            actual_yields=actual_yields,
            submitted_yields=submitted_yields,
            benchmark_yields=benchmark_yields,
        )
        session.add(sub)
        await session.commit()
        await session.refresh(sub)
    else:
        cow_metadata = await _load_challenge_cow_metadata(challenge, session, storage)
        actual_yields = await _load_challenge_actual_yields(challenge, session, storage)
        submitted_yields, benchmark_yields = await _load_submission_yields(sub, session, storage)

    parities = {cid: meta.get("parity", 1) or 1 for cid, meta in cow_metadata.items()}
    pdf_bytes = generate_report_pdf(
        stats=sub.stats,
        challenger_yields=submitted_yields,
        benchmark_yields=benchmark_yields,
        actual_yields=actual_yields,
        parities=parities,
        challenge_name=challenge.name or challenge.dataset,
        challenge_source="reference dataset"
        if challenge.dataset == "icar" and challenge.source == "preset"
        else challenge.source or challenge.dataset,
        submission_type=sub.submission_type,
        challenger_label=sub.model_type or "own method",
        benchmark_label=sub.benchmark_model or "tim",
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=benchmark_report_{submission_id}.pdf"
        },
    )
