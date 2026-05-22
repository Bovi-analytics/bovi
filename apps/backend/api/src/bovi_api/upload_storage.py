"""Raw upload storage helpers for audit-ready CSV ingestion."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient, ContentSettings

from bovi_api.auth import CurrentUser
from bovi_api.models import UploadAudit
from bovi_api.settings import Settings


class UploadStorageError(RuntimeError):
    """Raised when a raw upload cannot be persisted before processing."""


@dataclass(frozen=True)
class StoredUpload:
    """Metadata returned after raw bytes are written to Blob Storage."""

    original_filename: str
    content_type: str | None
    size_bytes: int
    sha256: str
    blob_container: str
    blob_path: str


def _safe_filename(filename: str) -> str:
    name = Path(filename or "upload.csv").name.strip() or "upload.csv"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:180]


def build_upload_blob_path(
    action_type: str,
    filename: str,
    *,
    upload_id: str | None = None,
    now: datetime | None = None,
) -> str:
    """Build the canonical upload blob path."""
    timestamp = now or datetime.now(timezone.utc)
    upload_key = upload_id or str(uuid.uuid4())
    return f"uploads/{timestamp:%Y/%m/%d}/{upload_key}/{action_type}/{_safe_filename(filename)}"


def upload_csv_to_blob(
    content: bytes,
    *,
    filename: str,
    content_type: str | None,
    action_type: str,
    settings: Settings,
) -> StoredUpload:
    """Persist raw CSV bytes in Azure Blob Storage before parsing."""
    if not settings.connection_string:
        raise UploadStorageError(
            "Azure Blob Storage is not configured (CONNECTION_STRING missing)."
        )

    digest = hashlib.sha256(content).hexdigest()
    blob_path = build_upload_blob_path(action_type, filename)
    try:
        service = BlobServiceClient.from_connection_string(settings.connection_string)
        blob = service.get_blob_client(settings.upload_container, blob_path)
        blob.upload_blob(
            content,
            overwrite=False,
            content_settings=ContentSettings(content_type=content_type or "text/csv"),
            metadata={
                "action_type": action_type,
                "original_filename": _safe_filename(filename),
                "sha256": digest,
                "size_bytes": str(len(content)),
            },
        )
    except AzureError as exc:
        raise UploadStorageError("Could not persist uploaded CSV to Azure Blob Storage.") from exc

    return StoredUpload(
        original_filename=filename or "upload.csv",
        content_type=content_type,
        size_bytes=len(content),
        sha256=digest,
        blob_container=settings.upload_container,
        blob_path=blob_path,
    )


def make_upload_audit(
    stored: StoredUpload,
    *,
    action_type: str,
    status: str,
    current_user: CurrentUser,
    organization_id: int | None,
    organization_name: str | None,
    challenge_id: int | None = None,
    submission_id: int | None = None,
    error_detail: str | None = None,
) -> UploadAudit:
    """Create an UploadAudit row from stored blob metadata and auth context."""
    return UploadAudit(
        action_type=action_type,
        status=status,
        user_id=current_user.id,
        user_email=current_user.email,
        user_name=current_user.name,
        organization_id=organization_id,
        organization_name=organization_name,
        original_filename=stored.original_filename,
        content_type=stored.content_type,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        blob_container=stored.blob_container,
        blob_path=stored.blob_path,
        challenge_id=challenge_id,
        submission_id=submission_id,
        error_detail=error_detail,
    )


def organization_name_for_user(
    current_user: CurrentUser, organization_id: int | None
) -> str | None:
    """Return the current user's organization name for audit snapshots."""
    if organization_id is None:
        return None
    for organization in current_user.organizations:
        if organization.id == organization_id:
            return organization.name
    return None
