"""Shared pytest fixtures for the bovi-api test suite."""

import asyncio
import os

import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ["DATABASE_URL"] = ""

from bovi_api.app import create_app
from bovi_api.database import get_session
from bovi_api.models import Challenge, FittingResult, HerdProfile, Submission


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
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _create_tables() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(FittingResult.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(HerdProfile.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(Challenge.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(Submission.__table__.create)  # type: ignore[union-attr]

    asyncio.run(_create_tables())

    async def override_get_session():
        async with session_factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_session] = override_get_session

    yield ASGITestClient(app)
