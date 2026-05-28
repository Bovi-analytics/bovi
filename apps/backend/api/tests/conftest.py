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
from bovi_api.auth import AuthenticatedOrganization, CurrentUser, require_auth
from bovi_api.database import get_session
from bovi_api.models import (
    Challenge,
    FittingResult,
    HerdProfile,
    Organization,
    OrganizationInvite,
    OrganizationMembership,
    StorageArtifact,
    Submission,
    UploadedDataset,
    User,
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


@dataclass(frozen=True)
class AuthFixtureUser:
    id: int
    entra_tenant_id: str
    entra_oid: str
    email: str
    name: str
    organizations: tuple[AuthenticatedOrganization, ...]
    roles: tuple[str, ...] = ("User",)
    is_admin: bool = False

    def current_user(self) -> CurrentUser:
        return CurrentUser(
            id=self.id,
            entra_tenant_id=self.entra_tenant_id,
            entra_oid=self.entra_oid,
            account_type="entra",
            email=self.email,
            name=self.name,
            roles=list(self.roles),
            is_admin=self.is_admin,
            organizations=list(self.organizations),
        )


class MultiOrgAuth:
    def __init__(self, client: "ASGITestClient", users: dict[str, AuthFixtureUser]) -> None:
        self.client = client
        self.users = users

    def as_user(self, key: str) -> None:
        user = self.users[key]

        async def override_require_auth() -> CurrentUser:
            return user.current_user()

        self.client.app.dependency_overrides[require_auth] = override_require_auth


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

    def patch(self, path: str, **kwargs) -> httpx.Response:
        return asyncio.run(self._request("PATCH", path, **kwargs))

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
            await conn.run_sync(User.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(Organization.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(OrganizationMembership.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(OrganizationInvite.__table__.create)  # type: ignore[union-attr]
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

    async def override_require_auth():
        return CurrentUser(
            id=1,
            entra_tenant_id="test-tenant",
            entra_oid="test-user-oid",
            account_type="entra",
            email="user@example.test",
            name="Test User",
            roles=["User"],
            organizations=[AuthenticatedOrganization(id=1, name="Test Organization", role="Owner")],
        )

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[require_auth] = override_require_auth
    app.dependency_overrides[get_artifact_storage] = lambda: artifact_storage
    app.dependency_overrides[get_optional_artifact_storage] = lambda: artifact_storage
    app.state.blob_container_client = container_client

    async def _seed_auth_rows() -> None:
        async with session_factory() as session:
            user = User(
                id=1,
                entra_tenant_id="test-tenant",
                entra_oid="test-user-oid",
                account_type="entra",
                email="user@example.test",
                name="Test User",
            )
            organization = Organization(id=1, name="Test Organization")
            session.add(user)
            session.add(organization)
            await session.commit()
            membership = OrganizationMembership(user_id=1, organization_id=1, role="Owner")
            session.add(membership)
            await session.commit()

    asyncio.run(_seed_auth_rows())

    yield ASGITestClient(app)


@pytest.fixture()
def multi_org_auth(client: ASGITestClient) -> MultiOrgAuth:
    """Seed a reusable two-organization auth scenario and switch active users."""
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> None:
        async for session in override():
            org_two = Organization(id=2, name="Other Organization")
            orgless = User(
                id=3,
                entra_tenant_id="test-tenant",
                entra_oid="orgless-user-oid",
                account_type="entra",
                email="orgless@example.test",
                name="Orgless User",
            )
            user_two = User(
                id=2,
                entra_tenant_id="test-tenant",
                entra_oid="other-user-oid",
                account_type="entra",
                email="other@example.test",
                name="Other User",
            )
            admin = User(
                id=7,
                entra_tenant_id="admin-tenant",
                entra_oid="admin-oid",
                account_type="entra",
                email="admin@example.test",
                name="Admin User",
                role="Admin",
            )
            session.add(org_two)
            session.add(user_two)
            session.add(orgless)
            session.add(admin)
            await session.commit()
            session.add(OrganizationMembership(user_id=2, organization_id=2, role="Owner"))
            await session.commit()
            break

    asyncio.run(_seed())

    users = {
        "org1_owner": AuthFixtureUser(
            id=1,
            entra_tenant_id="test-tenant",
            entra_oid="test-user-oid",
            email="user@example.test",
            name="Test User",
            organizations=(
                AuthenticatedOrganization(id=1, name="Test Organization", role="Owner"),
            ),
        ),
        "org2_owner": AuthFixtureUser(
            id=2,
            entra_tenant_id="test-tenant",
            entra_oid="other-user-oid",
            email="other@example.test",
            name="Other User",
            organizations=(
                AuthenticatedOrganization(id=2, name="Other Organization", role="Owner"),
            ),
        ),
        "orgless": AuthFixtureUser(
            id=3,
            entra_tenant_id="test-tenant",
            entra_oid="orgless-user-oid",
            email="orgless@example.test",
            name="Orgless User",
            organizations=(),
        ),
        "admin": AuthFixtureUser(
            id=7,
            entra_tenant_id="admin-tenant",
            entra_oid="admin-oid",
            email="admin@example.test",
            name="Admin User",
            organizations=(),
            roles=("Admin",),
            is_admin=True,
        ),
    }
    auth = MultiOrgAuth(client, users)
    auth.as_user("org1_owner")
    return auth
