"""Helpers for attaching user and organization display metadata to API records."""

from __future__ import annotations

from typing import Any, TypeVar

from sqlmodel import SQLModel

from bovi_api.models import Organization, User

ResponseModel = TypeVar("ResponseModel", bound=SQLModel)


def owner_fields(user: User | None, organization: Organization | None) -> dict[str, Any]:
    """Return the common display fields used by organization-scoped resources."""
    return {
        "user_name": user.name if user is not None else None,
        "user_email": user.email if user is not None else None,
        "organization_name": organization.name if organization is not None else None,
    }


def read_model(
    record: SQLModel,
    schema: type[ResponseModel],
    user: User | None = None,
    organization: Organization | None = None,
    **extra: Any,
) -> ResponseModel:
    """Build a response schema from a DB record plus owner display fields."""
    data = record.model_dump()
    data.update(owner_fields(user, organization))
    data.update(extra)
    return schema.model_validate(data)
