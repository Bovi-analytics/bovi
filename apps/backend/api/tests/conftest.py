"""Shared pytest fixtures for the bovi-api test suite."""

import asyncio

import pytest
from bovi_api import app as app_module
from bovi_api.app import create_app
from bovi_api.database import get_session
from bovi_api.models import Challenge, HerdProfile, Submission
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture()
def client(monkeypatch):
    """TestClient backed by an in-memory SQLite database."""
    # Skip the Alembic auto-migration on startup: tests create tables directly
    # against the in-memory engine below, so running migrations would only
    # touch an unrelated real DB file.
    monkeypatch.setattr(app_module, "_run_migrations", lambda: None)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _create_tables() -> None:
        async with engine.begin() as conn:
            # Only create the tables actually used in tests - FittingResult uses
            # sa_type=None which SQLite cannot compile DDL for (works fine on
            # PostgreSQL in production)
            await conn.run_sync(HerdProfile.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(Challenge.__table__.create)  # type: ignore[union-attr]
            await conn.run_sync(Submission.__table__.create)  # type: ignore[union-attr]

    asyncio.run(_create_tables())

    async def override_get_session():
        async with session_factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_session] = override_get_session

    with TestClient(app) as c:
        yield c
