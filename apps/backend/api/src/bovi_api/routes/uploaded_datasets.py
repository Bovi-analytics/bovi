"""Organization-scoped access to persisted uploaded dataset metadata."""

from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from bovi_api.auth import CurrentUser, ensure_organization_access, require_auth
from bovi_api.database import get_session
from bovi_api.models import (
    Organization,
    StorageArtifact,
    UploadedDataset,
    UploadedDatasetDetail,
    UploadedDatasetRead,
    User,
)
from bovi_api.ownership import read_model
from bovi_api.storage import (
    ArtifactStorage,
    delete_artifacts_best_effort,
    delete_unreferenced_artifacts,
    get_artifact_storage,
    load_json_artifact,
)

router = APIRouter(
    prefix="/uploaded-datasets",
    tags=["uploaded-datasets"],
    dependencies=[Depends(require_auth)],
)


async def _uploaded_dataset_row(
    session: AsyncSession,
    dataset_id: str,
) -> tuple[UploadedDataset, User | None, Organization | None] | None:
    result = await session.execute(
        select(UploadedDataset, User, Organization)
        .outerjoin(User, col(UploadedDataset.user_id) == col(User.id))
        .outerjoin(Organization, col(UploadedDataset.organization_id) == col(Organization.id))
        .where(UploadedDataset.id == dataset_id)
    )
    return cast(
        tuple[UploadedDataset, User | None, Organization | None] | None, result.one_or_none()
    )


@router.get("", response_model=list[UploadedDatasetRead], include_in_schema=False)
@router.get("/", response_model=list[UploadedDatasetRead])
async def list_uploaded_datasets(
    organization_id: str | None = None,
    scope: Literal["organization", "mine"] = "organization",
    user_id: int | None = None,
    sort: Literal["uploaded_at", "name", "user"] = "uploaded_at",
    direction: Literal["asc", "desc"] = "desc",
    q: str | None = None,
    current_user: CurrentUser = Depends(require_auth),
    session: AsyncSession = Depends(get_session),
) -> list[UploadedDatasetRead]:
    """List uploaded datasets visible in the selected organization."""
    statement = (
        select(UploadedDataset, User, Organization)
        .outerjoin(User, col(UploadedDataset.user_id) == col(User.id))
        .outerjoin(Organization, col(UploadedDataset.organization_id) == col(Organization.id))
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
        statement = statement.where(UploadedDataset.organization_id == selected_organization_id)
    elif current_user.is_admin:
        pass
    else:
        raise HTTPException(status_code=422, detail="organization_id is required.")

    if user_id is not None:
        statement = statement.where(UploadedDataset.user_id == user_id)
    elif scope == "mine":
        statement = statement.where(UploadedDataset.user_id == current_user.id)
    if q:
        statement = statement.where(col(UploadedDataset.name).contains(q))
    sort_column = {
        "uploaded_at": col(UploadedDataset.uploaded_at),
        "name": col(UploadedDataset.name),
        "user": col(User.name),
    }[sort]
    statement = statement.order_by(sort_column.asc() if direction == "asc" else sort_column.desc())
    statement = statement.limit(100)
    result = await session.execute(statement)
    return [
        read_model(dataset, UploadedDatasetRead, user, organization)
        for dataset, user, organization in result.all()
    ]


@router.get("/{dataset_id}", response_model=UploadedDatasetDetail)
async def get_uploaded_dataset(
    dataset_id: str,
    current_user: CurrentUser = Depends(require_auth),
    session: AsyncSession = Depends(get_session),
    storage: ArtifactStorage = Depends(get_artifact_storage),
) -> UploadedDatasetDetail:
    """Retrieve uploaded dataset metadata and parsed cows/stats for reuse."""
    row = await _uploaded_dataset_row(session, dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Uploaded dataset not found")
    dataset, user, organization = row
    ensure_organization_access(current_user, dataset.organization_id)
    cows = await load_json_artifact(
        session=session,
        storage=storage,
        artifact_id=dataset.cows_artifact_id,
    )
    stats_payload = await load_json_artifact(
        session=session,
        storage=storage,
        artifact_id=dataset.stats_artifact_id,
    )
    if not isinstance(cows, list):
        cows = []
    stats = stats_payload if isinstance(stats_payload, dict) else {}
    return read_model(
        dataset,
        UploadedDatasetDetail,
        user,
        organization,
        cows=cows,
        stats=stats.get("stats", dataset.stats_summary),
        raw_stats=stats.get("raw_stats", dataset.raw_stats_summary),
    )


@router.delete("/{dataset_id}", status_code=204)
async def delete_uploaded_dataset(
    dataset_id: str,
    current_user: CurrentUser = Depends(require_auth),
    session: AsyncSession = Depends(get_session),
    storage: ArtifactStorage = Depends(get_artifact_storage),
) -> None:
    """Delete uploaded dataset metadata and best-effort delete its blobs."""
    dataset = await session.get(UploadedDataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Uploaded dataset not found")
    ensure_organization_access(current_user, dataset.organization_id)

    artifact_ids = [
        dataset.raw_file_artifact_id,
        dataset.cows_artifact_id,
        dataset.stats_artifact_id,
    ]
    artifacts = [
        artifact
        for artifact_id in artifact_ids
        if artifact_id is not None
        if (artifact := await session.get(StorageArtifact, artifact_id)) is not None
    ]
    await session.delete(dataset)
    await session.flush()
    artifacts_to_delete = await delete_unreferenced_artifacts(
        session=session,
        artifacts=artifacts,
    )
    await session.commit()
    await delete_artifacts_best_effort(storage, artifacts_to_delete)
