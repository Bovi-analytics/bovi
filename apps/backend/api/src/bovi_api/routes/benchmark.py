"""Benchmark endpoints — ICAR accreditation workflow."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from bovi_api.benchmark_ingestion import parse_submission_csv
from bovi_api.benchmark_stats import calculate_comparison_stats
from bovi_api.database import get_session
from bovi_api.models import (
    Challenge,
    ChallengeDetail,
    ChallengeRead,
    Submission,
    SubmissionRead,
)
from bovi_api.routes.datasets import DatasetKey, PeriodKey, SizeKey, fetch_preset_cows
from bovi_api.routes.proxy import _get_client
from bovi_api.settings import Settings, get_settings

logger = logging.getLogger("bovi_api.benchmark")
router = APIRouter(prefix="/benchmark", tags=["benchmark"])

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_FAILURE_THRESHOLD = 0.20  # >20% failed cows → reject submission


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _call_tim(
    cow_metadata: dict[str, dict],
    settings: Settings,
) -> dict[str, float]:
    """Call POST /curves/test-interval for all cows in cow_metadata.

    Args:
        cow_metadata: {cow_id: {parity, dim: list[int], milk_kg: list[float]}}
        settings: App settings (for lactation_curves_url).

    Returns:
        {cow_id: yield_305day}

    Raises:
        HTTPException 502 if upstream unreachable.

    """
    # Build flat arrays with test_ids to identify each measurement
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
            f"{settings.lactation_curves_url}/test-interval",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
    except httpx.RequestError as exc:
        logger.exception("TIM proxy error: %s", exc)
        raise HTTPException(status_code=502, detail="Upstream lactation-curves service unavailable.")

    data = resp.json()
    # The Function App returns [{test_id, yield_305day}, ...] or {test_id: yield}
    # Normalise both formats to {cow_id: float}
    if isinstance(data, list):
        return {item["test_id"]: item["yield_305day"] for item in data}
    # dict format
    return {str(k): float(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Challenge endpoints
# ---------------------------------------------------------------------------


class ChallengeCreateBody(BaseModel):
    dataset: DatasetKey
    size: Literal["small", "medium"]  # "large" is out of scope (timeout risk)
    period: PeriodKey


@router.post("/challenges", response_model=ChallengeRead, status_code=201)
async def create_challenge(
    body: ChallengeCreateBody,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Challenge:
    """Create a challenge by sampling cows from a preset dataset and computing reference yields."""
    # fetch_preset_cows is synchronous (Azure Blob SDK); run in executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    preset = await loop.run_in_executor(
        None, fetch_preset_cows, body.dataset, body.size, body.period, settings
    )

    # Build cow_metadata from PresetCow objects
    cow_metadata: dict[str, dict] = {
        cow.cow_id: {
            "parity": cow.parity,
            "dim": cow.dim,
            "milk_kg": cow.milk_kg,
        }
        for cow in preset.cows
    }

    # Compute reference yields via TIM
    reference_yields = await _call_tim(cow_metadata, settings)

    challenge = Challenge(
        dataset=body.dataset,
        size=body.size,
        period=body.period,
        cow_metadata=cow_metadata,
        reference_yields=reference_yields,
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
        select(Challenge).order_by(Challenge.created_at.desc()).limit(100)
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
    """Download a CSV of the challenge's test-day cow data for Pad B users."""
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


class SubmissionBodviModel(BaseModel):
    submission_type: str = "bovi_model"
    model_type: str = "tim"
    organization: str | None = None
    country: str | None = None
    notes: str | None = None


@router.post("/challenges/{challenge_id}/submissions", response_model=SubmissionRead, status_code=201)
async def create_submission_bovi(
    challenge_id: int,
    body: SubmissionBodviModel,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Submission:
    """Pad A: compute 305-day yields for all challenge cows using a bovi model, then submit."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")

    if body.model_type == "tim":
        bovi_yields = await _call_tim(challenge.cow_metadata, settings)
    else:
        raise HTTPException(
            status_code=422,
            detail=f"model_type '{body.model_type}' not supported yet. Use 'tim'.",
        )

    parities = {cid: meta.get("parity", 1) or 1 for cid, meta in challenge.cow_metadata.items()}
    stats = calculate_comparison_stats(bovi_yields, challenge.reference_yields, parities)

    submission = Submission(
        challenge_id=challenge_id,
        submission_type="bovi_model",
        model_type=body.model_type,
        organization=body.organization,
        country=body.country,
        submitted_yields=bovi_yields,
        bovi_yields=bovi_yields,
        stats=stats,
        failed_cow_ids=[],
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
    organization: str | None = Form(default=None),
    country: str | None = Form(default=None),
    calculation_method: str | None = Form(default=None),
    notes: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Submission:
    """Pad B: upload a CSV of own-method yields, compare against reference."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")

    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")

    try:
        submitted_yields, failed_ids = parse_submission_csv(content, return_failed=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    total = len(submitted_yields) + len(failed_ids)
    if total > 0 and len(failed_ids) / total > _FAILURE_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail=f"Too many invalid rows: {len(failed_ids)}/{total} failed (>{int(_FAILURE_THRESHOLD*100)}% threshold).",
        )

    # Auto-compute bovi TIM yields for report flavors 2 and 3
    bovi_yields = await _call_tim(challenge.cow_metadata, settings)

    parities = {cid: meta.get("parity", 1) or 1 for cid, meta in challenge.cow_metadata.items()}
    stats = calculate_comparison_stats(submitted_yields, challenge.reference_yields, parities)

    submission = Submission(
        challenge_id=challenge_id,
        submission_type="own_method",
        organization=organization,
        country=country,
        calculation_method=calculation_method,
        notes=notes,
        submitted_yields=submitted_yields,
        bovi_yields=bovi_yields,
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
        select(Submission).order_by(Submission.created_at.desc()).limit(100)
    )
    return list(result.scalars().all())


@router.get("/submissions/{submission_id}", response_model=SubmissionRead)
async def get_submission(
    submission_id: int,
    session: AsyncSession = Depends(get_session),
) -> Submission:
    """Get a single submission with stats."""
    sub = await session.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    return sub
