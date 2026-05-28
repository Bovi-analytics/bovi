"""Persistence endpoints for storing and retrieving fitting results."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from bovi_api.auth import (
    CurrentUser,
    ensure_organization_access,
    first_accessible_organization_id,
    require_auth,
)
from bovi_api.database import get_session
from bovi_api.models import (
    FittingResult,
    FittingResultCreate,
    FittingResultRead,
    Organization,
    User,
)
from bovi_api.ownership import read_model

router = APIRouter(prefix="/results", tags=["results"], dependencies=[Depends(require_auth)])


@router.post("/", response_model=FittingResultRead)
async def store_result(
    result: FittingResultCreate,
    organization_id: int | None = None,
    current_user: CurrentUser = Depends(require_auth),
    session: AsyncSession = Depends(get_session),
) -> FittingResultRead:
    """Store a fitting/prediction result."""
    selected_organization_id = organization_id or first_accessible_organization_id(current_user)
    ensure_organization_access(current_user, selected_organization_id)
    db_result = FittingResult.model_validate(
        result,
        update={"user_id": current_user.id, "organization_id": selected_organization_id},
    )
    session.add(db_result)
    await session.commit()
    await session.refresh(db_result)
    return await _fitting_result_read(session, db_result.id or 0)


@router.get("/{result_id}", response_model=FittingResultRead)
async def get_result(
    result_id: int,
    current_user: CurrentUser = Depends(require_auth),
    session: AsyncSession = Depends(get_session),
) -> FittingResultRead:
    """Retrieve a stored result by ID."""
    result = await session.get(FittingResult, result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    ensure_organization_access(current_user, result.organization_id)
    return await _fitting_result_read(session, result_id)


@router.get("/", response_model=list[FittingResultRead])
async def list_results(
    organization_id: str | None = None,
    scope: str = "organization",
    user_id: int | None = None,
    model_type: str | None = None,
    source_app: str | None = None,
    limit: int = 50,
    offset: int = 0,
    current_user: CurrentUser = Depends(require_auth),
    session: AsyncSession = Depends(get_session),
) -> list[FittingResultRead]:
    """List stored results with optional filtering."""
    query = (
        select(FittingResult, User, Organization)
        .outerjoin(User, col(FittingResult.user_id) == col(User.id))
        .outerjoin(Organization, col(FittingResult.organization_id) == col(Organization.id))
    )

    if organization_id == "all":
        if not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required.")
    elif organization_id is not None:
        try:
            selected_organization_id = int(organization_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="organization_id must be an integer or all"
            ) from exc
        ensure_organization_access(current_user, selected_organization_id)
        query = query.where(FittingResult.organization_id == selected_organization_id)
    elif current_user.is_admin:
        pass
    else:
        selected_organization_id = first_accessible_organization_id(current_user)
        query = query.where(FittingResult.organization_id == selected_organization_id)

    if user_id is not None:
        query = query.where(FittingResult.user_id == user_id)
    elif scope == "mine":
        query = query.where(FittingResult.user_id == current_user.id)

    if model_type:
        query = query.where(FittingResult.model_type == model_type)
    if source_app:
        query = query.where(FittingResult.source_app == source_app)

    query = query.order_by(col(FittingResult.created_at).desc()).offset(offset).limit(limit)
    result = await session.execute(query)
    return [
        read_model(fitting_result, FittingResultRead, user, organization)
        for fitting_result, user, organization in result.all()
    ]


async def _fitting_result_read(session: AsyncSession, result_id: int) -> FittingResultRead:
    result = await session.execute(
        select(FittingResult, User, Organization)
        .outerjoin(User, col(FittingResult.user_id) == col(User.id))
        .outerjoin(Organization, col(FittingResult.organization_id) == col(Organization.id))
        .where(FittingResult.id == result_id)
    )
    fitting_result, user, organization = result.one()
    return read_model(fitting_result, FittingResultRead, user, organization)
