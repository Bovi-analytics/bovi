"""CRUD endpoints for user-managed herd stat profiles."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from bovi_api.database import get_session
from bovi_api.models import HerdProfile, HerdProfileCreate, HerdProfileRead

router = APIRouter(tags=["herd-profiles"])


@router.get("/", response_model=list[HerdProfileRead])
async def list_herd_profiles(
    session: AsyncSession = Depends(get_session),
) -> list[HerdProfile]:
    """List all herd profiles, newest first."""
    result = await session.execute(
        select(HerdProfile).order_by(HerdProfile.created_at.desc())
    )
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
