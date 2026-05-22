"""Shared pytest fixtures for the bovi-api test suite."""
# ruff: noqa: E402, I001

import asyncio
import os

import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ["DATABASE_URL"] = ""

from bovi_api.auth import AuthenticatedOrganization, CurrentUser, require_auth
from bovi_api.app import create_app
from bovi_api.database import get_session
from bovi_api.models import (
    Challenge,
    FittingResult,
    HerdProfile,
    Organization,
    OrganizationInvite,
    OrganizationMembership,
    Submission,
    UploadAudit,
    User,
)
from bovi_api.routes import benchmark as benchmark_routes
from bovi_api.routes import herd_profiles as herd_profile_routes
from bovi_api.upload_storage import StoredUpload


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

    def _fake_upload_csv_to_blob(content, *, filename, content_type, action_type, settings):
        return StoredUpload(
            original_filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            sha256="0" * 64,
            blob_container=settings.upload_container,
            blob_path=f"test-uploads/{action_type}/{filename}",
        )

    monkeypatch.setattr(benchmark_routes, "upload_csv_to_blob", _fake_upload_csv_to_blob)
    monkeypatch.setattr(herd_profile_routes, "upload_csv_to_blob", _fake_upload_csv_to_blob)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _create_tables() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(User.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(Organization.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(OrganizationMembership.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(OrganizationInvite.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(FittingResult.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(HerdProfile.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(Challenge.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(Submission.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(UploadAudit.__table__.create)  # type: ignore[union-attr]

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
            membership = OrganizationMembership(user_id=1, organization_id=1, role="Owner")
            session.add(user)
            session.add(organization)
            session.add(membership)
            await session.commit()

    asyncio.run(_seed_auth_rows())

    yield ASGITestClient(app)
