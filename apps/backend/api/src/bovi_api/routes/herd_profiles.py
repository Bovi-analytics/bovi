"""CRUD endpoints for user-managed herd stat profiles."""

import json
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from bovi_api.database import get_session
from bovi_api.herd_stats_ingestion import DEFAULT_STAT_RANGES, normalize_herd_stats, parse_csv
from bovi_api.models import HerdProfile, HerdProfileCreate, HerdProfileRead, UploadedDataset
from bovi_api.settings import Settings, get_settings
from bovi_api.storage import (
    ArtifactStorage,
    create_bytes_artifact,
    create_json_artifact,
    delete_artifacts_best_effort,
    get_optional_artifact_storage,
)

router = APIRouter(tags=["herd-profiles"])

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class CowRecordPayload(BaseModel):
    """Per-cow test-day records for client-side reuse in the Curves tab."""

    cow_id: str
    parity: int | None
    dim: list[int]
    milk_kg: list[float]


class HerdProfileUploadResponse(BaseModel):
    """Preview of normalized herd stats parsed from a CSV upload."""

    upload_id: str | None = None
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
    session: AsyncSession = Depends(get_session),
    storage: ArtifactStorage | None = Depends(get_optional_artifact_storage),
) -> HerdProfileUploadResponse:
    """Parse, normalize, and optionally persist an uploaded CSV preview."""
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
    upload_id: str | None = None
    if storage is not None:
        upload_id = str(uuid4())
        prefix = storage.path("herd-datasets", upload_id)
        cows_payload = [
            {
                "cow_id": c.cow_id,
                "parity": c.parity,
                "dim": c.dim,
                "milk_kg": c.milk_kg,
            }
            for c in (result.cows or [])
        ]
        uploaded = [
            await create_bytes_artifact(
                session=session,
                storage=storage,
                artifact_kind="herd_dataset_csv",
                entity_type="uploaded_dataset",
                entity_uuid=upload_id,
                blob_path=f"{prefix}/raw/upload.csv",
                data=content,
                content_type="text/csv",
                original_filename=filename,
                row_count=result.row_count,
                record_count=result.cow_count,
            ),
            await create_json_artifact(
                session=session,
                storage=storage,
                artifact_kind="herd_dataset_cows_json",
                entity_type="uploaded_dataset",
                entity_uuid=upload_id,
                blob_path=f"{prefix}/parsed/cows.json.gz",
                payload=cows_payload,
                row_count=result.row_count,
                record_count=result.cow_count,
            ),
            await create_json_artifact(
                session=session,
                storage=storage,
                artifact_kind="herd_dataset_stats_json",
                entity_type="uploaded_dataset",
                entity_uuid=upload_id,
                blob_path=f"{prefix}/parsed/stats.json.gz",
                payload={"stats": normalized, "raw_stats": result.raw_stats},
                row_count=result.row_count,
                record_count=result.cow_count,
            ),
        ]
        dataset = UploadedDataset(
            id=upload_id,
            name=filename,
            dataset_type="herd_profile",
            format_detected=result.format_detected,
            raw_file_artifact_id=uploaded[0].id,
            cows_artifact_id=uploaded[1].id,
            stats_artifact_id=uploaded[2].id,
            original_filename=filename,
            row_count=result.row_count,
            cow_count=result.cow_count,
            detected_parity=result.detected_parity,
            columns=result.columns or [],
            column_mapping=result.column_mapping or {},
            warnings=result.warnings,
            stats_summary=normalized,
            raw_stats_summary=result.raw_stats,
        )
        session.add(dataset)
        try:
            await session.commit()
        except Exception:
            await session.rollback()
            await delete_artifacts_best_effort(storage, uploaded)
            raise

    return HerdProfileUploadResponse(
        upload_id=upload_id,
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
