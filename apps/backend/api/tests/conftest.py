"""Shared pytest fixtures for the bovi-api test suite."""

import asyncio
import os
from dataclasses import dataclass

import httpx
import pytest
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from bovi_core.storage import BlobStore
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ["DATABASE_URL"] = ""

from bovi_api.app import create_app
from bovi_api.database import get_session
from bovi_api.models import (
    Challenge,
    FittingResult,
    HerdProfile,
    StorageArtifact,
    Submission,
    UploadedDataset,
)
from bovi_api.storage import ArtifactStorage, get_artifact_storage, get_optional_artifact_storage


@dataclass
class _StoredBlob:
    data: bytes
    metadata: dict[str, str]
    content_type: str | None
    content_encoding: str | None
    etag: str


class _Download:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class _BlobClient:
    def __init__(self, store: dict[str, _StoredBlob], path: str) -> None:
        self._store = store
        self._path = path

    def upload_blob(self, data, *, overwrite, metadata, content_settings) -> None:
        if not overwrite and self._path in self._store:
            raise ResourceExistsError(message="exists")
        payload = bytes(data)
        self._store[self._path] = _StoredBlob(
            data=payload,
            metadata=metadata,
            content_type=content_settings.content_type,
            content_encoding=content_settings.content_encoding,
            etag=f"etag-{len(self._store) + 1}",
        )

    def get_blob_properties(self) -> dict:
        if self._path not in self._store:
            raise ResourceNotFoundError(message="missing")
        return {"etag": self._store[self._path].etag}

    def download_blob(self, **kwargs) -> _Download:
        if self._path not in self._store:
            raise ResourceNotFoundError(message="missing")
        data = self._store[self._path].data
        offset = kwargs.get("offset")
        length = kwargs.get("length")
        if offset is not None:
            data = data[offset : offset + length if length is not None else None]
        return _Download(data)

    def exists(self) -> bool:
        return self._path in self._store

    def delete_blob(self, **_kwargs) -> bool:
        if self._path not in self._store:
            raise ResourceNotFoundError(message="missing")
        del self._store[self._path]
        return True


class _ContainerClient:
    def __init__(self) -> None:
        self.store: dict[str, _StoredBlob] = {}

    def get_blob_client(self, *, blob: str) -> _BlobClient:
        return _BlobClient(self.store, blob)


class ASGITestClient:
    """Small sync wrapper around httpx ASGITransport for local app tests."""

    def __init__(self, app) -> None:
        self.app = app

    def get(self, path: str, **kwargs) -> httpx.Response:
        return asyncio.run(self._request("GET", path, **kwargs))

    def post(self, path: str, **kwargs) -> httpx.Response:
        return asyncio.run(self._request("POST", path, **kwargs))

    def put(self, path: str, **kwargs) -> httpx.Response:
        return asyncio.run(self._request("PUT", path, **kwargs))

    def delete(self, path: str, **kwargs) -> httpx.Response:
        return asyncio.run(self._request("DELETE", path, **kwargs))

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=30,
        ) as client:
            return await client.request(method, path, **kwargs)


@pytest.fixture()
def client(monkeypatch):
    """TestClient backed by an in-memory SQLite database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    container_client = _ContainerClient()
    artifact_storage = ArtifactStorage(
        BlobStore(container_client, account_name="testaccount", container_name="testcontainer"),
        prefix="bovi/uploads",
        environment="test",
    )

    async def _create_tables() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(FittingResult.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(HerdProfile.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(StorageArtifact.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(UploadedDataset.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(Challenge.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(Submission.__table__.create)  # type: ignore[union-attr]

    asyncio.run(_create_tables())

    async def override_get_session():
        async with session_factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_artifact_storage] = lambda: artifact_storage
    app.dependency_overrides[get_optional_artifact_storage] = lambda: artifact_storage
    app.state.blob_container_client = container_client

    yield ASGITestClient(app)
