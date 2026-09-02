"""Authentication endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from bovi_api.auth import CurrentUser, current_user_payload, require_auth
from bovi_api.database import get_session
from bovi_api.models import TermsAcceptanceAudit
from bovi_api.settings import Settings, get_settings
from bovi_api.terms import (
    CURRENT_TERMS_DOCUMENT_FILENAME,
    CURRENT_TERMS_DOCUMENT_SHA256,
    CURRENT_TERMS_KEY,
    CURRENT_TERMS_VERSION,
    get_terms_acceptance_status,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/status")
async def auth_status(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    """Return auth configuration status for the dashboard."""
    configured = bool(settings.azure_ad_client_id)
    return {
        "auth_configured": configured,
        "require_auth": not settings.auth_disabled,
        "auth_mode": "disabled" if settings.auth_disabled else "azure_ad",
    }


@router.get("/me")
async def me(
    current_user: Annotated[CurrentUser, Depends(require_auth)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, Any]:
    """Return the current local Bovi user and organizations."""
    payload = await current_user_payload(current_user)
    payload["terms_acceptance"] = (
        await get_terms_acceptance_status(current_user.id, session)
    ).model_dump()
    return payload


@router.post("/terms/accept")
async def accept_terms(
    request: Request,
    current_user: Annotated[CurrentUser, Depends(require_auth)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, Any]:
    """Record that the current user accepted the active Terms document."""
    result = await session.execute(
        select(TermsAcceptanceAudit)
        .where(TermsAcceptanceAudit.user_id == current_user.id)
        .where(TermsAcceptanceAudit.terms_key == CURRENT_TERMS_KEY)
        .where(TermsAcceptanceAudit.terms_version == CURRENT_TERMS_VERSION)
        .where(TermsAcceptanceAudit.document_sha256 == CURRENT_TERMS_DOCUMENT_SHA256)
    )
    existing = result.scalar_one_or_none()
    if existing is None:
        audit = TermsAcceptanceAudit(
            user_id=current_user.id,
            terms_key=CURRENT_TERMS_KEY,
            terms_version=CURRENT_TERMS_VERSION,
            document_sha256=CURRENT_TERMS_DOCUMENT_SHA256,
            document_filename=CURRENT_TERMS_DOCUMENT_FILENAME,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        session.add(audit)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
    return (await get_terms_acceptance_status(current_user.id, session)).model_dump()


@router.get("/verify")
async def verify(current_user: Annotated[CurrentUser, Depends(require_auth)]) -> dict[str, Any]:
    """Verify the current token."""
    return {"valid": True, "user": await current_user_payload(current_user)}
