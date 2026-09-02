"""Terms of Use version metadata and acceptance helpers."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from bovi_api.models import TermsAcceptanceAudit

CURRENT_TERMS_KEY = "terms-of-use-data-contribution"
CURRENT_TERMS_VERSION = "072326"
CURRENT_TERMS_EFFECTIVE_DATE = "2026-07-09"
CURRENT_TERMS_DOCUMENT_FILENAME = "Terms of Use and Data Contribution Agreement 072326.docx"
CURRENT_TERMS_DOCUMENT_SHA256 = (
    "dba8cbba07f6a413d868bfccc4b671f974b48335cc1b5ca2677a73e1ce758304"  # pragma: allowlist secret
)
CURRENT_TERMS_DOCUMENT_URL = "/legal/terms-of-use-data-contribution-agreement-072326.docx"


class TermsAcceptanceStatus(BaseModel):
    """Current user's acceptance state for the active Terms document."""

    accepted: bool
    terms_key: str = CURRENT_TERMS_KEY
    terms_version: str = CURRENT_TERMS_VERSION
    document_sha256: str = CURRENT_TERMS_DOCUMENT_SHA256
    document_filename: str = CURRENT_TERMS_DOCUMENT_FILENAME
    document_url: str = CURRENT_TERMS_DOCUMENT_URL
    accepted_at: datetime | None = None


async def get_terms_acceptance_status(
    user_id: int,
    session: AsyncSession,
) -> TermsAcceptanceStatus:
    """Return whether a user accepted the current Terms document."""
    result = await session.execute(
        select(TermsAcceptanceAudit)
        .where(TermsAcceptanceAudit.user_id == user_id)
        .where(TermsAcceptanceAudit.terms_key == CURRENT_TERMS_KEY)
        .where(TermsAcceptanceAudit.terms_version == CURRENT_TERMS_VERSION)
        .where(TermsAcceptanceAudit.document_sha256 == CURRENT_TERMS_DOCUMENT_SHA256)
        .order_by(col(TermsAcceptanceAudit.accepted_at).desc())
    )
    audit = result.scalars().first()
    return TermsAcceptanceStatus(
        accepted=audit is not None,
        accepted_at=audit.accepted_at if audit else None,
    )
