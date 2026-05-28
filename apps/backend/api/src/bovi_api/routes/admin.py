"""Admin-only overview endpoints."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from bovi_api.auth import CurrentUser, require_admin
from bovi_api.database import get_session
from bovi_api.models import Challenge, HerdProfile, Organization, Submission, UploadedDataset, User

router = APIRouter(prefix="/admin", tags=["admin"])

AdminDataCategory = Literal[
    "benchmark_submission",
    "benchmark_challenge",
    "herd_dataset_upload",
    "herd_profile",
]
AdminCategoryFilter = Literal[
    "all",
    "benchmark_submission",
    "benchmark_challenge",
    "herd_dataset_upload",
    "herd_profile",
]
AdminOverviewSort = Literal["created_at", "organization", "user", "category", "status"]
SortDirection = Literal["asc", "desc"]

CATEGORY_LABELS: dict[AdminDataCategory, str] = {
    "benchmark_submission": "Benchmark submissions",
    "benchmark_challenge": "Benchmark challenges",
    "herd_dataset_upload": "Herd dataset uploads",
    "herd_profile": "Herd profiles",
}


class AdminOverviewItem(BaseModel):
    """Single admin feed item with metadata only."""

    item_type: AdminDataCategory
    item_type_label: str
    id: str
    numeric_id: int | None = None
    challenge_id: int | None = None
    organization_id: int | None = None
    organization_name: str | None = None
    user_id: int | None = None
    user_email: str | None = None
    user_name: str | None = None
    title: str
    created_at: datetime | None = None
    status: str
    source: str | None = None
    submission_type: str | None = None
    benchmark_model: str | None = None
    row_count: int | None = None
    cow_count: int | None = None
    failed_count: int = 0
    primary_metric_label: str | None = None
    primary_metric_value: float | None = None


class AdminOrganizationBreakdown(BaseModel):
    """Counts and latest activity for one organization."""

    organization_id: int | None = None
    organization_name: str
    user_count: int = 0
    total_items: int = 0
    benchmark_submissions: int = 0
    benchmark_challenges: int = 0
    herd_dataset_uploads: int = 0
    herd_profiles: int = 0
    failed_items: int = 0
    latest_activity_at: datetime | None = None


class AdminCategoryBreakdown(BaseModel):
    """Counts and latest activity for one data category."""

    item_type: AdminDataCategory
    label: str
    count: int = 0
    failed_count: int = 0
    latest_activity_at: datetime | None = None


class AdminOverviewKpis(BaseModel):
    """Top-level admin overview counters."""

    total_items: int = 0
    organizations: int = 0
    users: int = 0
    benchmark_submissions: int = 0
    benchmark_challenges: int = 0
    herd_dataset_uploads: int = 0
    herd_profiles: int = 0
    failed_items: int = 0
    latest_activity_at: datetime | None = None


class AdminOverviewResponse(BaseModel):
    """Read-only admin overview payload."""

    kpis: AdminOverviewKpis
    by_organization: list[AdminOrganizationBreakdown]
    by_category: list[AdminCategoryBreakdown]
    items: list[AdminOverviewItem]


def _latest(a: datetime | None, b: datetime | None) -> datetime | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _safe_int(value: int | None) -> int | None:
    return value if value is not None else None


def _metric_from_stats(stats: dict | None) -> tuple[str | None, float | None]:
    if not isinstance(stats, dict):
        return None, None
    candidates = (
        stats.get("challenger_vs_aly"),
        stats.get("overall"),
        stats.get("vs_aly", {}).get("overall") if isinstance(stats.get("vs_aly"), dict) else None,
    )
    overall = next(
        (
            block.get("overall") if isinstance(block, dict) and "overall" in block else block
            for block in candidates
            if isinstance(block, dict)
        ),
        None,
    )
    if not isinstance(overall, dict):
        return None, None
    rmse = overall.get("rmse")
    if isinstance(rmse, int | float):
        return "RMSE", float(rmse)
    pearson = overall.get("pearson")
    if isinstance(pearson, int | float):
        return "Pearson", float(pearson)
    return None, None


def _status_from_upload(dataset: UploadedDataset) -> str:
    return "warning" if dataset.warnings else "ready"


def _is_problem(item: AdminOverviewItem) -> bool:
    return item.failed_count > 0 or item.status not in {"ready", "completed"}


def _matches_text(item: AdminOverviewItem, q: str | None) -> bool:
    if not q:
        return True
    needle = q.casefold()
    values = [
        item.title,
        item.item_type_label,
        item.organization_name,
        item.user_email,
        item.user_name,
        item.status,
        item.source,
        item.submission_type,
        item.benchmark_model,
    ]
    return any(needle in value.casefold() for value in values if value)


def _matches_date(
    item: AdminOverviewItem, created_from: datetime | None, created_to: datetime | None
) -> bool:
    if item.created_at is None:
        return created_from is None and created_to is None
    if created_from is not None and item.created_at < created_from:
        return False
    if created_to is not None and item.created_at > created_to:
        return False
    return True


def _sort_key(item: AdminOverviewItem, sort: AdminOverviewSort) -> tuple:
    if sort == "organization":
        return (item.organization_name or "", item.created_at or datetime.min)
    if sort == "user":
        return (item.user_email or item.user_name or "", item.created_at or datetime.min)
    if sort == "category":
        return (item.item_type_label, item.created_at or datetime.min)
    if sort == "status":
        return (item.status, item.created_at or datetime.min)
    return (item.created_at or datetime.min,)


def _category_count_attr(category: AdminDataCategory) -> str:
    return {
        "benchmark_submission": "benchmark_submissions",
        "benchmark_challenge": "benchmark_challenges",
        "herd_dataset_upload": "herd_dataset_uploads",
        "herd_profile": "herd_profiles",
    }[category]


async def _fetch_items(session: AsyncSession) -> list[AdminOverviewItem]:
    items: list[AdminOverviewItem] = []

    submission_rows = await session.execute(
        select(Submission, Organization, User)
        .join(Organization, col(Submission.organization_id) == col(Organization.id), isouter=True)
        .join(User, col(Submission.user_id) == col(User.id), isouter=True)
    )
    for submission, organization, user in submission_rows.all():
        metric_label, metric_value = _metric_from_stats(submission.stats)
        title = (
            submission.organization
            or submission.calculation_method
            or submission.model_type
            or "Benchmark submission"
        )
        items.append(
            AdminOverviewItem(
                item_type="benchmark_submission",
                item_type_label=CATEGORY_LABELS["benchmark_submission"],
                id=str(submission.id or ""),
                numeric_id=_safe_int(submission.id),
                challenge_id=submission.challenge_id,
                organization_id=submission.organization_id,
                organization_name=organization.name if organization else None,
                user_id=submission.user_id,
                user_email=user.email if user else None,
                user_name=user.name if user else None,
                title=title,
                created_at=submission.created_at,
                status=submission.ingest_status,
                source=submission.calculation_method or "benchmark",
                submission_type=submission.submission_type,
                benchmark_model=submission.benchmark_model,
                row_count=submission.row_count,
                cow_count=submission.submitted_yield_count,
                failed_count=submission.failed_count,
                primary_metric_label=metric_label,
                primary_metric_value=metric_value,
            )
        )

    challenge_rows = await session.execute(
        select(Challenge, Organization, User)
        .join(Organization, col(Challenge.organization_id) == col(Organization.id), isouter=True)
        .join(User, col(Challenge.user_id) == col(User.id), isouter=True)
    )
    for challenge, organization, user in challenge_rows.all():
        items.append(
            AdminOverviewItem(
                item_type="benchmark_challenge",
                item_type_label=CATEGORY_LABELS["benchmark_challenge"],
                id=str(challenge.id or ""),
                numeric_id=_safe_int(challenge.id),
                organization_id=challenge.organization_id,
                organization_name=organization.name if organization else None,
                user_id=challenge.user_id,
                user_email=user.email if user else None,
                user_name=user.name if user else None,
                title=challenge.name or f"{challenge.dataset} challenge",
                created_at=challenge.created_at,
                status=challenge.ingest_status,
                source=challenge.source or challenge.dataset,
                row_count=challenge.row_count,
                cow_count=challenge.cow_count,
                failed_count=0,
            )
        )

    dataset_rows = await session.execute(
        select(UploadedDataset, Organization, User)
        .join(
            Organization,
            col(UploadedDataset.organization_id) == col(Organization.id),
            isouter=True,
        )
        .join(User, col(UploadedDataset.user_id) == col(User.id), isouter=True)
    )
    for dataset, organization, user in dataset_rows.all():
        items.append(
            AdminOverviewItem(
                item_type="herd_dataset_upload",
                item_type_label=CATEGORY_LABELS["herd_dataset_upload"],
                id=dataset.id,
                organization_id=dataset.organization_id,
                organization_name=organization.name if organization else None,
                user_id=dataset.user_id,
                user_email=user.email if user else None,
                user_name=user.name if user else None,
                title=dataset.name or dataset.original_filename,
                created_at=dataset.uploaded_at,
                status=_status_from_upload(dataset),
                source=dataset.format_detected,
                row_count=dataset.row_count,
                cow_count=dataset.cow_count,
                failed_count=len(dataset.warnings),
            )
        )

    profile_rows = await session.execute(
        select(HerdProfile, Organization, User)
        .join(
            Organization,
            col(HerdProfile.organization_id) == col(Organization.id),
            isouter=True,
        )
        .join(User, col(HerdProfile.user_id) == col(User.id), isouter=True)
    )
    for profile, organization, user in profile_rows.all():
        items.append(
            AdminOverviewItem(
                item_type="herd_profile",
                item_type_label=CATEGORY_LABELS["herd_profile"],
                id=str(profile.id or ""),
                numeric_id=_safe_int(profile.id),
                organization_id=profile.organization_id,
                organization_name=organization.name if organization else None,
                user_id=profile.user_id,
                user_email=user.email if user else None,
                user_name=user.name if user else None,
                title=profile.name,
                created_at=profile.created_at,
                status="ready",
                source="herd profile",
                failed_count=0,
            )
        )

    return items


def _build_response(items: list[AdminOverviewItem], limit: int) -> AdminOverviewResponse:
    organization_map: dict[int | None, AdminOrganizationBreakdown] = {}
    organization_users: dict[int | None, set[int]] = defaultdict(set)
    category_map = {
        category: AdminCategoryBreakdown(item_type=category, label=label)
        for category, label in CATEGORY_LABELS.items()
    }
    latest_activity_at: datetime | None = None
    failed_items = 0

    for item in items:
        category_attr = _category_count_attr(item.item_type)
        org_key = item.organization_id
        if org_key not in organization_map:
            organization_map[org_key] = AdminOrganizationBreakdown(
                organization_id=item.organization_id,
                organization_name=item.organization_name or "Unknown organization",
            )
        org = organization_map[org_key]
        org.total_items += 1
        setattr(org, category_attr, getattr(org, category_attr) + 1)
        org.latest_activity_at = _latest(org.latest_activity_at, item.created_at)
        if item.user_id is not None:
            organization_users[org_key].add(item.user_id)

        category = category_map[item.item_type]
        category.count += 1
        category.latest_activity_at = _latest(category.latest_activity_at, item.created_at)

        latest_activity_at = _latest(latest_activity_at, item.created_at)
        if _is_problem(item):
            failed_items += 1
            org.failed_items += 1
            category.failed_count += 1

    for org_key, org in organization_map.items():
        org.user_count = len(organization_users[org_key])

    kpis = AdminOverviewKpis(
        total_items=len(items),
        organizations=len({item.organization_id for item in items if item.organization_id}),
        users=len({item.user_id for item in items if item.user_id}),
        benchmark_submissions=category_map["benchmark_submission"].count,
        benchmark_challenges=category_map["benchmark_challenge"].count,
        herd_dataset_uploads=category_map["herd_dataset_upload"].count,
        herd_profiles=category_map["herd_profile"].count,
        failed_items=failed_items,
        latest_activity_at=latest_activity_at,
    )

    return AdminOverviewResponse(
        kpis=kpis,
        by_organization=sorted(
            organization_map.values(),
            key=lambda org: (org.latest_activity_at or datetime.min, org.organization_name),
            reverse=True,
        ),
        by_category=list(category_map.values()),
        items=items[:limit],
    )


@router.get("/submissions-overview", response_model=AdminOverviewResponse)
async def submissions_overview(
    current_user: Annotated[CurrentUser, Depends(require_admin)],
    session: Annotated[AsyncSession, Depends(get_session)],
    organization_id: str | None = Query(default=None),
    category: AdminCategoryFilter = "all",
    user_id: int | None = None,
    q: str | None = None,
    created_from: datetime | None = Query(default=None, alias="from"),
    created_to: datetime | None = Query(default=None, alias="to"),
    sort: AdminOverviewSort = "created_at",
    direction: SortDirection = "desc",
    limit: int = Query(default=100, ge=1, le=500),
) -> AdminOverviewResponse:
    """Return a read-only cross-organization admin overview."""
    _ = current_user
    items = await _fetch_items(session)

    if organization_id and organization_id != "all":
        try:
            selected_organization_id = int(organization_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="organization_id must be an integer or all"
            ) from exc
        items = [item for item in items if item.organization_id == selected_organization_id]
    if category != "all":
        items = [item for item in items if item.item_type == category]
    if user_id is not None:
        items = [item for item in items if item.user_id == user_id]
    items = [
        item
        for item in items
        if _matches_text(item, q) and _matches_date(item, created_from, created_to)
    ]
    items = sorted(items, key=lambda item: _sort_key(item, sort), reverse=direction == "desc")

    return _build_response(items, limit)
