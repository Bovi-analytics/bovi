"""Integration tests for benchmark routes."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from bovi_api.app import create_app
from bovi_api.database import get_session

DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


def test_list_challenges_empty(client):
    """GET /benchmark/challenges returns empty list when no challenges exist."""
    import asyncio

    async def _run():
        response = await client.get("/benchmark/challenges")
        assert response.status_code == 200
        assert response.json() == []

    asyncio.run(_run())


def test_get_submission_not_found(client):
    """GET /benchmark/submissions/999 → 404."""
    import asyncio
    async def _run():
        resp = await client.get("/benchmark/submissions/999")
        assert resp.status_code == 404
    asyncio.run(_run())


def test_list_submissions_empty(client):
    """GET /benchmark/submissions → []."""
    import asyncio
    async def _run():
        resp = await client.get("/benchmark/submissions")
        assert resp.status_code == 200
        assert resp.json() == []
    asyncio.run(_run())


def test_pad_b_upload_rejects_high_failure_rate(client):
    """POST upload with >20% bad rows → 422."""
    import asyncio
    # 10 rows total, 9 invalid → 90% failure rate → should be rejected
    csv_content = b"cow_id,yield_305day\n" + b"".join(
        f"cow{i},bad_value\n".encode() for i in range(9)
    ) + b"cow9,8000.0\n"
    async def _run():
        resp = await client.post(
            "/benchmark/challenges/1/submissions/upload",
            files={"file": ("results.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 422
    asyncio.run(_run())
