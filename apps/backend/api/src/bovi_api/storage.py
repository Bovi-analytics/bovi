"""Blob-backed artifact storage for the central API."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from typing import Any
from uuid import uuid4

from bovi_core.storage import BlobStore
from fastapi import Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select
from starlette.concurrency import run_in_threadpool

from bovi_api.models import Challenge, StorageArtifact, Submission, UploadedDataset
from bovi_api.settings import Settings, get_settings

_SAFE_PATH_PART = re.compile(r"[^A-Za-z0-9_.=-]+")


class ArtifactStorage:
    """Service object that writes large API payloads to blob storage."""

    def __init__(self, store: BlobStore, *, prefix: str, environment: str) -> None:
        self.store = store
        self.prefix = _strip_slashes(prefix)
        self.environment = _safe_path_part(environment or "local")

    def path(self, *parts: str) -> str:
        clean = [_safe_path_part(part) for part in parts if part]
        return "/".join([self.prefix, self.environment, *clean])


def get_artifact_storage(settings: Settings = Depends(get_settings)) -> ArtifactStorage:
    """FastAPI dependency for upload artifact storage."""
    if not (
        settings.storage_account_name_icar
        and settings.storage_account_key_icar
        and settings.storage_account_container_icar
    ):
        raise HTTPException(
            status_code=503,
            detail="ICAR blob storage is not configured for uploads.",
        )
    store = BlobStore.from_connection_parts(
        account_name=settings.storage_account_name_icar,
        account_key=settings.storage_account_key_icar,
        container_name=settings.storage_account_container_icar,
    )
    return ArtifactStorage(
        store,
        prefix=settings.upload_blob_prefix,
        environment=settings.bovi_env,
    )


def get_optional_artifact_storage(
    settings: Settings = Depends(get_settings),
) -> ArtifactStorage | None:
    """Return artifact storage when configured, otherwise None for legacy reads."""
    if not (
        settings.storage_account_name_icar
        and settings.storage_account_key_icar
        and settings.storage_account_container_icar
    ):
        return None
    return get_artifact_storage(settings)


async def create_bytes_artifact(
    *,
    session: AsyncSession,
    storage: ArtifactStorage,
    artifact_kind: str,
    entity_type: str,
    entity_uuid: str,
    blob_path: str,
    data: bytes,
    content_type: str,
    content_encoding: str | None = None,
    original_filename: str | None = None,
    row_count: int | None = None,
    record_count: int | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> StorageArtifact:
    """Upload bytes and add a StorageArtifact row to the session.

    Existing artifacts with identical stored bytes are reused to avoid storing
    duplicate uploads under different entity-specific paths.
    """
    sha256 = hashlib.sha256(data).hexdigest()
    existing = await _find_existing_artifact(
        session=session,
        storage=storage,
        sha256=sha256,
        size_bytes=len(data),
        content_type=content_type,
        content_encoding=content_encoding,
    )
    if existing is not None:
        return existing

    artifact_id = str(uuid4())
    metadata = _blob_metadata(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        entity_type=entity_type,
        entity_uuid=entity_uuid,
    )
    result = await run_in_threadpool(
        storage.store.upload_bytes,
        blob_path,
        data,
        content_type=content_type,
        content_encoding=content_encoding,
        metadata=metadata,
        overwrite=False,
    )
    artifact = StorageArtifact(
        id=artifact_id,
        artifact_kind=artifact_kind,
        entity_type=entity_type,
        entity_uuid=entity_uuid,
        storage_account_name=storage.store.account_name,
        container_name=storage.store.container_name,
        blob_path=result.blob_path,
        original_filename=original_filename,
        content_type=result.content_type,
        content_encoding=result.content_encoding,
        size_bytes=result.size_bytes,
        sha256=result.sha256,
        etag=result.etag,
        row_count=row_count,
        record_count=record_count,
        schema_version=1,
        metadata_extra=metadata_extra or {},
    )
    session.add(artifact)
    return artifact


async def create_json_artifact(
    *,
    session: AsyncSession,
    storage: ArtifactStorage,
    artifact_kind: str,
    entity_type: str,
    entity_uuid: str,
    blob_path: str,
    payload: Any,
    row_count: int | None = None,
    record_count: int | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> StorageArtifact:
    """Upload a gzip-compressed JSON payload and add a StorageArtifact row."""
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    data = gzip.compress(raw, mtime=0)
    return await create_bytes_artifact(
        session=session,
        storage=storage,
        artifact_kind=artifact_kind,
        entity_type=entity_type,
        entity_uuid=entity_uuid,
        blob_path=blob_path,
        data=data,
        content_type="application/json",
        content_encoding="gzip",
        original_filename=None,
        row_count=row_count,
        record_count=record_count,
        metadata_extra=metadata_extra,
    )


async def load_json_artifact(
    *,
    session: AsyncSession,
    storage: ArtifactStorage,
    artifact_id: str | None,
) -> Any:
    """Load a JSON artifact by id."""
    if artifact_id is None:
        return None
    artifact = await session.get(StorageArtifact, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=500, detail=f"Storage artifact not found: {artifact_id}")
    return await run_in_threadpool(storage.store.download_json_gzip, artifact.blob_path)


async def load_bytes_artifact(
    *,
    session: AsyncSession,
    storage: ArtifactStorage,
    artifact_id: str | None,
) -> bytes | None:
    """Load a bytes artifact by id."""
    if artifact_id is None:
        return None
    artifact = await session.get(StorageArtifact, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=500, detail=f"Storage artifact not found: {artifact_id}")
    return await run_in_threadpool(storage.store.download_bytes, artifact.blob_path)


async def delete_artifacts_best_effort(
    storage: ArtifactStorage,
    artifacts: list[StorageArtifact],
    *,
    session: AsyncSession | None = None,
) -> None:
    """Best-effort cleanup for blobs uploaded before a failed DB commit.

    When a session is provided, artifacts that still have committed DB rows are
    treated as pre-existing/reused and are not deleted.
    """
    for artifact in _unique_artifacts(artifacts):
        if session is not None and await session.get(StorageArtifact, artifact.id) is not None:
            continue
        await run_in_threadpool(storage.store.delete_if_exists, artifact.blob_path)


async def delete_unreferenced_artifacts(
    *,
    session: AsyncSession,
    artifacts: list[StorageArtifact],
) -> list[StorageArtifact]:
    """Delete artifact rows when no API record still references them."""
    removable: list[StorageArtifact] = []
    for artifact in _unique_artifacts(artifacts):
        if await _artifact_has_references(session=session, artifact_id=artifact.id):
            continue
        stored = await session.get(StorageArtifact, artifact.id)
        if stored is None:
            continue
        await session.delete(stored)
        removable.append(stored)

    await session.flush()
    return removable


def _blob_metadata(
    *,
    artifact_id: str,
    artifact_kind: str,
    entity_type: str,
    entity_uuid: str,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_kind": artifact_kind,
        "entity_type": entity_type,
        "entity_uuid": entity_uuid,
        "created_by_app": "bovi-api",
        "schema_version": "1",
    }


async def _find_existing_artifact(
    *,
    session: AsyncSession,
    storage: ArtifactStorage,
    sha256: str,
    size_bytes: int,
    content_type: str,
    content_encoding: str | None,
) -> StorageArtifact | None:
    result = await session.execute(
        select(StorageArtifact)
        .where(StorageArtifact.storage_account_name == storage.store.account_name)
        .where(StorageArtifact.container_name == storage.store.container_name)
        .where(StorageArtifact.sha256 == sha256)
        .where(StorageArtifact.size_bytes == size_bytes)
        .where(StorageArtifact.content_type == content_type)
        .where(StorageArtifact.content_encoding == content_encoding)
        .order_by(col(StorageArtifact.created_at).asc())
        .limit(1)
    )
    return result.scalars().first()


async def _artifact_has_references(*, session: AsyncSession, artifact_id: str) -> bool:
    checks = [
        select(UploadedDataset.id)
        .where(
            or_(
                col(UploadedDataset.raw_file_artifact_id) == artifact_id,
                col(UploadedDataset.cows_artifact_id) == artifact_id,
                col(UploadedDataset.stats_artifact_id) == artifact_id,
            )
        )
        .limit(1),
        select(Challenge.id)
        .where(
            or_(
                col(Challenge.test_day_file_artifact_id) == artifact_id,
                col(Challenge.actual_yields_file_artifact_id) == artifact_id,
                col(Challenge.cow_metadata_artifact_id) == artifact_id,
                col(Challenge.actual_yields_artifact_id) == artifact_id,
            )
        )
        .limit(1),
        select(Submission.id)
        .where(
            or_(
                col(Submission.input_file_artifact_id) == artifact_id,
                col(Submission.submitted_yields_artifact_id) == artifact_id,
                col(Submission.bovi_yields_artifact_id) == artifact_id,
            )
        )
        .limit(1),
    ]
    for statement in checks:
        result = await session.execute(statement)
        if result.first() is not None:
            return True
    return False


def _unique_artifacts(artifacts: list[StorageArtifact]) -> list[StorageArtifact]:
    unique: dict[str, StorageArtifact] = {}
    for artifact in artifacts:
        unique.setdefault(artifact.id, artifact)
    return list(unique.values())


def _strip_slashes(value: str) -> str:
    return value.strip("/")


def _safe_path_part(value: str) -> str:
    clean = _SAFE_PATH_PART.sub("-", value.strip()).strip("-")
    return clean or "unnamed"
