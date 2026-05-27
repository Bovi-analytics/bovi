"""CRUD endpoints for user-managed herd stat profiles."""

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from bovi_api.database import get_session
from bovi_api.herd_stats_ingestion import DEFAULT_STAT_RANGES, normalize_herd_stats, parse_csv
from bovi_api.models import HerdProfile, HerdProfileCreate, HerdProfileRead
from bovi_api.settings import Settings, get_settings

router = APIRouter(tags=["herd-profiles"])

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class CowRecordPayload(BaseModel):
    """Per-cow test-day records for client-side reuse in the Curves tab."""

    cow_id: str
    parity: int | None
    dim: list[int]
    milk_kg: list[float]


class HerdProfileUploadResponse(BaseModel):
    """Preview of normalized herd stats parsed from a CSV upload. Not saved to DB."""

    stats: dict[str, float]
    raw_stats: dict[str, float]
    format_detected: str
    row_count: int
    warnings: list[str]
    cow_count: int | None = None
    detected_parity: int | None = None
    cows: list[CowRecordPayload] = []
    columns: list[str] = []
    column_mapping: dict[str, str] = {}
    mapping_required: bool = False


@router.get("", response_model=list[HerdProfileRead], include_in_schema=False)
@router.get("/", response_model=list[HerdProfileRead])
async def list_herd_profiles(
    session: AsyncSession = Depends(get_session),
) -> list[HerdProfile]:
    """List all herd profiles, newest first."""
    result = await session.execute(select(HerdProfile).order_by(col(HerdProfile.created_at).desc()))
    return list(result.scalars().all())


@router.post("", response_model=HerdProfileRead, status_code=201, include_in_schema=False)
@router.post("/", response_model=HerdProfileRead, status_code=201)
async def create_herd_profile(
    profile: HerdProfileCreate,
    session: AsyncSession = Depends(get_session),
) -> HerdProfile:
    """Create a new herd profile."""
    db_profile = HerdProfile(**profile.model_dump())
    session.add(db_profile)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=409, detail=f"A profile named '{profile.name}' already exists."
        )
    await session.refresh(db_profile)
    return db_profile


@router.post("/csv-preview", response_model=HerdProfileUploadResponse)
async def csv_preview(
    file: UploadFile = File(...),
    column_mapping: str | None = Form(default=None),
    settings: Settings = Depends(get_settings),
) -> HerdProfileUploadResponse:
    """Parse and normalize an uploaded CSV. Returns a preview; does NOT save to DB."""
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")

    parsed_mapping: dict[str, str] | None = None
    if column_mapping:
        try:
            decoded = json.loads(column_mapping)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid column mapping JSON.") from exc
        if not isinstance(decoded, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in decoded.items()
        ):
            raise HTTPException(status_code=400, detail="Column mapping must be a string map.")
        parsed_mapping = decoded

    try:
        result = parse_csv(
            content,
            allow_dairy_comp=settings.allow_dairy_comp_uploads,
            column_mapping=parsed_mapping,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized = normalize_herd_stats(result.raw_stats, DEFAULT_STAT_RANGES)

    return HerdProfileUploadResponse(
        stats=normalized,
        raw_stats=result.raw_stats,
        format_detected=result.format_detected,
        row_count=result.row_count,
        warnings=result.warnings,
        cow_count=result.cow_count,
        detected_parity=result.detected_parity,
        columns=result.columns or [],
        column_mapping=result.column_mapping or {},
        mapping_required=result.mapping_required,
        cows=[
            CowRecordPayload(
                cow_id=c.cow_id,
                parity=c.parity,
                dim=c.dim,
                milk_kg=c.milk_kg,
            )
            for c in (result.cows or [])
        ],
    )


@router.get("/{profile_id}", response_model=HerdProfileRead)
async def get_herd_profile(
    profile_id: int,
    session: AsyncSession = Depends(get_session),
) -> HerdProfile:
    """Retrieve a single herd profile by ID."""
    profile = await session.get(HerdProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Herd profile not found")
    return profile


@router.put("/{profile_id}", response_model=HerdProfileRead)
async def update_herd_profile(
    profile_id: int,
    update: HerdProfileCreate,
    session: AsyncSession = Depends(get_session),
) -> HerdProfile:
    """Update an existing herd profile."""
    profile = await session.get(HerdProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Herd profile not found")
    for field, value in update.model_dump().items():
        setattr(profile, field, value)
    session.add(profile)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=409, detail=f"A profile named '{update.name}' already exists."
        )
    await session.refresh(profile)
    return profile


@router.delete("/{profile_id}", status_code=204)
async def delete_herd_profile(
    profile_id: int,
    session: AsyncSession = Depends(get_session),
) -> None:
    """Delete a herd profile."""
    profile = await session.get(HerdProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Herd profile not found")
    await session.delete(profile)
    await session.commit()
