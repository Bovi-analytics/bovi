"""CRUD endpoints for user-managed herd stat profiles."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from bovi_api.auth import CurrentUser, first_accessible_organization_id, require_auth
from bovi_api.database import get_session
from bovi_api.herd_stats_ingestion import DEFAULT_STAT_RANGES, normalize_herd_stats, parse_csv
from bovi_api.models import HerdProfile, HerdProfileCreate, HerdProfileRead
from bovi_api.settings import Settings, get_settings
from bovi_api.upload_storage import (
    UploadStorageError,
    make_upload_audit,
    organization_name_for_user,
    upload_csv_to_blob,
)

router = APIRouter(tags=["herd-profiles"], dependencies=[Depends(require_auth)])

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ACTION_HERD_PROFILE_CSV_PREVIEW = "herd_profile_csv_preview"


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


@router.get("/", response_model=list[HerdProfileRead])
async def list_herd_profiles(
    session: AsyncSession = Depends(get_session),
) -> list[HerdProfile]:
    """List all herd profiles, newest first."""
    result = await session.execute(select(HerdProfile).order_by(col(HerdProfile.created_at).desc()))
    return list(result.scalars().all())


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
    current_user: CurrentUser = Depends(require_auth),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> HerdProfileUploadResponse:
    """Parse and normalize an uploaded CSV. Returns a preview; does NOT save to DB."""
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")

    organization_id = first_accessible_organization_id(current_user)
    organization_name = organization_name_for_user(current_user, organization_id)
    try:
        stored_upload = upload_csv_to_blob(
            content,
            filename=filename,
            content_type=file.content_type,
            action_type=ACTION_HERD_PROFILE_CSV_PREVIEW,
            settings=settings,
        )
    except UploadStorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = parse_csv(content)
    except ValueError as exc:
        session.add(
            make_upload_audit(
                stored_upload,
                action_type=ACTION_HERD_PROFILE_CSV_PREVIEW,
                status="rejected",
                current_user=current_user,
                organization_id=organization_id,
                organization_name=organization_name,
                error_detail=str(exc),
            )
        )
        await session.commit()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized = normalize_herd_stats(result.raw_stats, DEFAULT_STAT_RANGES)
    session.add(
        make_upload_audit(
            stored_upload,
            action_type=ACTION_HERD_PROFILE_CSV_PREVIEW,
            status="accepted",
            current_user=current_user,
            organization_id=organization_id,
            organization_name=organization_name,
        )
    )
    await session.commit()

    return HerdProfileUploadResponse(
        stats=normalized,
        raw_stats=result.raw_stats,
        format_detected=result.format_detected,
        row_count=result.row_count,
        warnings=result.warnings,
        cow_count=result.cow_count,
        detected_parity=result.detected_parity,
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
