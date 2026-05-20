"""Benchmark endpoints - ground-truth ALY benchmarking with selectable benchmark + challenger."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from typing import Literal

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

logger = logging.getLogger("bovi_api.benchmark")
router = APIRouter(prefix="/benchmark", tags=["benchmark"])

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_FAILURE_THRESHOLD = 0.20
_MAX_PLAUSIBLE_LACTATION_YIELD_KG = 100_000.0

CURVE_MODELS = {"wood", "wilmink", "ali_schaeffer", "fischer", "milkbot"}
YIELD_ESTIMATORS = {
    "tim": "/test-interval",
    "islc": "/islc",
    "best_predict": "/best-predict",
}
ALL_MODELS = CURVE_MODELS | set(YIELD_ESTIMATORS) | {"autoencoder"}

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
    """Refresh legacy ICAR preset challenges that stored bad concatenated ALY values."""
    if challenge.dataset != "icar" or challenge.source != "preset":
        return False
    if _actual_yields_are_plausible(challenge.actual_yields):
        return False

    loop = asyncio.get_running_loop()
    preset = await loop.run_in_executor(None, fetch_preset_cows, "icar", "full", "all", settings)
    if not preset.actual_yields:
        raise HTTPException(
            status_code=500,
            detail="ICAR preset is missing actual_yields - regenerate the preset data.",
        )

    challenge.cow_metadata = {
        cow.cow_id: {"parity": cow.parity, "dim": cow.dim, "milk_kg": cow.milk_kg}
        for cow in preset.cows
        if cow.cow_id in preset.actual_yields
    }
    challenge.actual_yields = {cid: preset.actual_yields[cid] for cid in challenge.cow_metadata}
    session.add(challenge)
    await session.commit()
    await session.refresh(challenge)
    return True


def _recalculate_submission_stats(submission: Submission, challenge: Challenge) -> dict:
    parities = {cid: meta.get("parity", 1) or 1 for cid, meta in challenge.cow_metadata.items()}
    return calculate_comparison_stats_v2(
        challenger_yields=submission.submitted_yields,
        benchmark_yields=submission.bovi_yields,
        actual_yields=challenge.actual_yields or {},
        parities=parities,
    )


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
) -> dict[str, float]:
    """Per-cow POST /characteristic - failed cows are omitted."""
    client = _get_client()
    url = f"{settings.lactation_curves_url}/characteristic"

    async def one(cow_id: str, meta: dict) -> tuple[str, float | None]:
        if len(meta.get("dim", [])) < 2:
            return cow_id, None
        payload = {
            "dim": meta["dim"],
            "milkrecordings": meta["milk_kg"],
            "model": model,
            "characteristic": "cumulative_milk_yield",
            "parity": meta.get("parity") or 1,
            "lactation_length": 305,
        }
        try:
            resp = await client.post(
                url,
                content=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            return cow_id, None
        data = resp.json()
        val = data.get("value")
        if val is None:
            return cow_id, None
        return cow_id, float(val)

    sem = asyncio.Semaphore(20)

    async def guarded(cow_id: str, meta: dict):
        async with sem:
            return await one(cow_id, meta)

    results = await asyncio.gather(*(guarded(cid, m) for cid, m in cow_metadata.items()))
    return {cid: val for cid, val in results if val is not None}


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
        items.append({"milk": milk, "parity": meta.get("parity") or 1})

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
) -> dict[str, float]:
    """Run the requested model on all cows and return {cow_id: 305-day yield}."""
    if model in YIELD_ESTIMATORS:
        return await _call_yield_estimator(cow_metadata, settings, YIELD_ESTIMATORS[model])
    if model in CURVE_MODELS:
        return await _call_curve_characteristic(cow_metadata, model, settings)
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


@router.post("/challenges", response_model=ChallengeRead, status_code=201)
async def create_challenge_preset(
    body: ChallengeCreatePresetBody,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Challenge:
    """Create a preset-backed challenge (ICAR cohort with ground-truth ALY)."""
    loop = asyncio.get_running_loop()
    preset = await loop.run_in_executor(
        None, fetch_preset_cows, body.preset, "full", "all", settings
    )
    if not preset.actual_yields:
        raise HTTPException(
            status_code=500,
            detail="ICAR preset is missing actual_yields - regenerate the preset blob.",
        )

    cow_metadata: dict[str, dict] = {
        cow.cow_id: {"parity": cow.parity, "dim": cow.dim, "milk_kg": cow.milk_kg}
        for cow in preset.cows
    }
    # Keep only cows that have ALY (defensive - already filtered by generator)
    cow_metadata = {cid: m for cid, m in cow_metadata.items() if cid in preset.actual_yields}

    challenge = Challenge(
        dataset=body.preset,
        size="full",
        period="all",
        source="preset",
        name=body.name or "ICAR cohort",
        cow_metadata=cow_metadata,
        reference_yields=None,
        actual_yields=preset.actual_yields,
    )
    session.add(challenge)
    await session.commit()
    await session.refresh(challenge)
    return challenge


@router.post("/challenges/upload", response_model=ChallengeRead, status_code=201)
async def create_challenge_upload(
    name: str = Form(...),
    test_day_csv: UploadFile = File(...),
    actual_yields_csv: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> Challenge:
    """Create an upload-backed challenge: user-supplied test-day records + ground-truth yields."""
    test_bytes = await test_day_csv.read()
    aly_bytes = await actual_yields_csv.read()
    if len(test_bytes) > _MAX_UPLOAD_BYTES or len(aly_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="One of the uploads exceeds the 10 MB limit.")

    try:
        cow_metadata = parse_test_day_csv(test_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"test-day CSV: {exc}") from exc
    try:
        actual_yields = parse_actual_yields_csv(aly_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"actual-yields CSV: {exc}") from exc

    # Coverage check: ALY must cover at least 80% of cows in the test-day file
    overlap = sum(1 for cid in cow_metadata if cid in actual_yields)
    if not cow_metadata or overlap / len(cow_metadata) < 0.80:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Actual-yields CSV must cover at least 80% of cows from the test-day CSV "
                f"(got {overlap}/{len(cow_metadata)})."
            ),
        )
    # Drop cows without ALY
    cow_metadata = {cid: m for cid, m in cow_metadata.items() if cid in actual_yields}

    challenge = Challenge(
        dataset="upload",
        size="custom",
        period="custom",
        source="upload",
        name=name,
        cow_metadata=cow_metadata,
        reference_yields=None,
        actual_yields={cid: actual_yields[cid] for cid in cow_metadata},
    )
    session.add(challenge)
    await session.commit()
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
    return list(result.scalars().all())


@router.get("/challenges/{challenge_id}", response_model=ChallengeDetail)
async def get_challenge(
    challenge_id: int,
    session: AsyncSession = Depends(get_session),
) -> Challenge:
    """Get a single challenge with full cow metadata."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    return challenge


@router.get("/challenges/{challenge_id}/export")
async def export_challenge(
    challenge_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Download a CSV of the challenge's test-day data."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["cow_id", "parity", "dim", "milk_kg"])
    for cow_id, meta in challenge.cow_metadata.items():
        parity = meta.get("parity", "")
        for d, m in zip(meta["dim"], meta["milk_kg"]):
            writer.writerow([cow_id, parity, d, m])

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
    organization: str | None = None
    country: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _challenger_differs(self) -> "SubmissionBoviBody":
        if self.challenger == self.benchmark:
            raise ValueError("Challenger and benchmark must differ.")
        return self


@router.post(
    "/challenges/{challenge_id}/submissions", response_model=SubmissionRead, status_code=201
)
async def create_submission_bovi(
    challenge_id: int,
    body: SubmissionBoviBody,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Submission:
    """Run challenger + benchmark models on the cohort and compare both vs ALY."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    if not challenge.actual_yields:
        raise HTTPException(
            status_code=422,
            detail="Challenge has no ground-truth ALY - cannot benchmark.",
        )

    challenger_yields, benchmark_yields = await asyncio.gather(
        _dispatch_model(body.challenger, challenge.cow_metadata, settings),
        _dispatch_model(body.benchmark, challenge.cow_metadata, settings),
    )

    parities = {cid: meta.get("parity", 1) or 1 for cid, meta in challenge.cow_metadata.items()}
    failed_cow_ids = [cid for cid in challenge.cow_metadata if cid not in challenger_yields]
    stats = calculate_comparison_stats_v2(
        challenger_yields=challenger_yields,
        benchmark_yields=benchmark_yields,
        actual_yields=challenge.actual_yields,
        parities=parities,
    )

    submission = Submission(
        challenge_id=challenge_id,
        submission_type="bovi_model",
        model_type=body.challenger,
        benchmark_model=body.benchmark,
        organization=body.organization,
        country=body.country,
        notes=body.notes,
        submitted_yields=challenger_yields,
        bovi_yields=benchmark_yields,
        stats=stats,
        failed_cow_ids=failed_cow_ids,
    )
    session.add(submission)
    await session.commit()
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
    organization: str | None = Form(default=None),
    country: str | None = Form(default=None),
    calculation_method: str | None = Form(default=None),
    notes: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Submission:
    """Upload a CSV of own-method 305-day yields; benchmark model runs server-side."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    if not challenge.actual_yields:
        raise HTTPException(
            status_code=422,
            detail="Challenge has no ground-truth ALY - cannot benchmark.",
        )
    if benchmark not in ALL_MODELS:
        raise HTTPException(status_code=422, detail=f"Unknown benchmark '{benchmark}'.")

    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")

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

    benchmark_yields = await _dispatch_model(benchmark, challenge.cow_metadata, settings)

    parities = {cid: meta.get("parity", 1) or 1 for cid, meta in challenge.cow_metadata.items()}
    stats = calculate_comparison_stats_v2(
        challenger_yields=challenger_yields,
        benchmark_yields=benchmark_yields,
        actual_yields=challenge.actual_yields,
        parities=parities,
    )

    submission = Submission(
        challenge_id=challenge_id,
        submission_type="own_method",
        model_type=None,
        benchmark_model=benchmark,
        organization=organization,
        country=country,
        calculation_method=calculation_method,
        notes=notes,
        submitted_yields=challenger_yields,
        bovi_yields=benchmark_yields,
        stats=stats,
        failed_cow_ids=failed_ids,
    )
    session.add(submission)
    await session.commit()
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
) -> Submission:
    """Get a single submission with stats."""
    sub = await session.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    challenge = await session.get(Challenge, sub.challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    if await _repair_preset_challenge_if_needed(challenge, session, settings):
        sub.stats = _recalculate_submission_stats(sub, challenge)
        session.add(sub)
        await session.commit()
        await session.refresh(sub)
    return sub


@router.get("/submissions/{submission_id}/report")
async def download_report(
    submission_id: int,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Download a PDF report for a submission."""
    sub = await session.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    challenge = await session.get(Challenge, sub.challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    if await _repair_preset_challenge_if_needed(challenge, session, settings):
        sub.stats = _recalculate_submission_stats(sub, challenge)
        session.add(sub)
        await session.commit()
        await session.refresh(sub)

    parities = {cid: meta.get("parity", 1) or 1 for cid, meta in challenge.cow_metadata.items()}
    pdf_bytes = generate_report_pdf(
        stats=sub.stats,
        challenger_yields=sub.submitted_yields,
        benchmark_yields=sub.bovi_yields,
        actual_yields=challenge.actual_yields or {},
        parities=parities,
        challenge_name=challenge.name or challenge.dataset,
        challenge_source=challenge.source or challenge.dataset,
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
