"""Organization-scoped access to persisted uploaded dataset metadata."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from bovi_api.auth import CurrentUser, ensure_organization_access, require_auth
from bovi_api.database import get_session
from bovi_api.models import (
    HerdProfile,
    Organization,
    UploadedDataset,
    UploadedDatasetDetail,
    UploadedDatasetRead,
    User,
)
from bovi_api.ownership import read_model
from bovi_api.storage import (
    ArtifactStorage,
    get_artifact_storage,
    load_json_artifact,
)

router = APIRouter(
    prefix="/uploaded-datasets",
    tags=["uploaded-datasets"],
    dependencies=[Depends(require_auth)],
)


class UploadedDatasetProfileReference(BaseModel):
    """Herd profile that references or appears to derive from an uploaded dataset."""

    id: int
    name: str
    user_name: str | None = None
    user_email: str | None = None
    reference_type: Literal["linked", "matching_stats"]


class UploadedDatasetDeleteImpact(BaseModel):
    """Objects impacted when archiving an uploaded dataset."""

    dataset_id: str
    dataset_name: str
    herd_profiles: list[UploadedDatasetProfileReference]


_PROFILE_STAT_TO_DATASET_STAT = {
    "achieved_21_milk": "Achieved21Milk",
    "achieved_305_milk": "Achieved305Milk",
    "achieved_75_milk": "Achieved75Milk",
    "achieved_milk": "AchievedMilk",
    "days_dry": "DaysDry",
    "days_in_milk": "DaysInMilk",
    "days_open": "DaysOpen",
    "days_pregnant": "DaysPregnant",
    "historic_calving_interval": "HistoricCalvingInterval",
    "quality_sequence": "QualitySequence",
}


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
    include_deleted: bool = False,
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
    if include_deleted:
        if not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required.")
    else:
        statement = statement.where(col(UploadedDataset.deleted_at).is_(None))
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
    if dataset.deleted_at is not None and not current_user.is_admin:
        raise HTTPException(status_code=404, detail="Uploaded dataset not found")
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


@router.get("/{dataset_id}/delete-impact", response_model=UploadedDatasetDeleteImpact)
async def get_uploaded_dataset_delete_impact(
    dataset_id: str,
    current_user: CurrentUser = Depends(require_auth),
    session: AsyncSession = Depends(get_session),
) -> UploadedDatasetDeleteImpact:
    """Return user-visible records related to an uploaded dataset before archiving it."""
    dataset = await session.get(UploadedDataset, dataset_id)
    if dataset is None or (dataset.deleted_at is not None and not current_user.is_admin):
        raise HTTPException(status_code=404, detail="Uploaded dataset not found")
    ensure_organization_access(current_user, dataset.organization_id)

    references = await _uploaded_dataset_profile_references(session, dataset)

    return UploadedDatasetDeleteImpact(
        dataset_id=dataset.id,
        dataset_name=dataset.name or dataset.original_filename,
        herd_profiles=[reference for reference, _profile in references],
    )


@router.delete("/{dataset_id}", status_code=204)
async def delete_uploaded_dataset(
    dataset_id: str,
    current_user: CurrentUser = Depends(require_auth),
    session: AsyncSession = Depends(get_session),
) -> None:
    """Archive uploaded dataset metadata and remove herd profiles derived from it."""
    dataset = await session.get(UploadedDataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Uploaded dataset not found")
    ensure_organization_access(current_user, dataset.organization_id)

    references = await _uploaded_dataset_profile_references(session, dataset)
    for _reference, profile in references:
        await session.delete(profile)

    if dataset.deleted_at is None:
        dataset.deleted_at = datetime.now(UTC)
        dataset.deleted_by_user_id = current_user.id
        session.add(dataset)
    await session.commit()


async def _uploaded_dataset_profile_references(
    session: AsyncSession,
    dataset: UploadedDataset,
) -> list[tuple[UploadedDatasetProfileReference, HerdProfile]]:
    result = await session.execute(
        select(HerdProfile, User)
        .outerjoin(User, col(HerdProfile.user_id) == col(User.id))
        .where(HerdProfile.organization_id == dataset.organization_id)
    )
    references: list[tuple[UploadedDatasetProfileReference, HerdProfile]] = []
    seen: set[int] = set()
    for profile, user in result.all():
        reference_type: Literal["linked", "matching_stats"] | None = None
        if profile.source_uploaded_dataset_id == dataset.id:
            reference_type = "linked"
        elif profile.source_uploaded_dataset_id is None and _profile_matches_dataset_stats(
            profile, dataset
        ):
            reference_type = "matching_stats"
        if reference_type is None or profile.id is None or profile.id in seen:
            continue
        seen.add(profile.id)
        references.append(
            (
                UploadedDatasetProfileReference(
                    id=profile.id,
                    name=profile.name,
                    user_name=user.name if user else None,
                    user_email=user.email if user else None,
                    reference_type=reference_type,
                ),
                profile,
            )
        )
    return references


def _profile_matches_dataset_stats(profile: HerdProfile, dataset: UploadedDataset) -> bool:
    if not isinstance(dataset.stats_summary, dict) or not dataset.stats_summary:
        return False
    for profile_field, stat_key in _PROFILE_STAT_TO_DATASET_STAT.items():
        expected = dataset.stats_summary.get(stat_key)
        actual = getattr(profile, profile_field)
        if not isinstance(expected, int | float) or abs(float(actual) - float(expected)) > 1e-9:
            return False
    return True
