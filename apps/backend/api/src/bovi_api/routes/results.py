"""Persistence endpoints for storing and retrieving fitting results."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from bovi_api.database import get_session
from bovi_api.models import FittingResult, FittingResultCreate, FittingResultRead

router = APIRouter(prefix="/results", tags=["results"])


@router.post("/", response_model=FittingResultRead)
async def store_result(
    result: FittingResultCreate,
    session: AsyncSession = Depends(get_session),
) -> FittingResult:
    """Store a fitting/prediction result."""
    db_result = FittingResult.model_validate(result)
    session.add(db_result)
    await session.commit()
    await session.refresh(db_result)
    return db_result


@router.get("/{result_id}", response_model=FittingResultRead)
async def get_result(
    result_id: int,
    session: AsyncSession = Depends(get_session),
) -> FittingResult:
    """Retrieve a stored result by ID."""
    result = await session.get(FittingResult, result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


@router.get("/", response_model=list[FittingResultRead])
async def list_results(
    model_type: str | None = None,
    source_app: str | None = None,
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
) -> list[FittingResult]:
    """List stored results with optional filtering."""
    query = select(FittingResult)

    if model_type:
        query = query.where(FittingResult.model_type == model_type)
    if source_app:
        query = query.where(FittingResult.source_app == source_app)

    query = query.order_by(FittingResult.created_at.desc()).offset(offset).limit(limit)
    results = await session.exec(query)
    return list(results.all())
