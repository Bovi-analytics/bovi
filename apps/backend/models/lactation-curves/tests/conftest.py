"""Shared fixtures for lactation-curves API tests.

Run tests:  uv run pytest
Or test against a deployed instance:
  API_BASE_URL=https://your-func.azurewebsites.net pytest
"""

import os
from collections.abc import Generator

import httpx
import pytest


@pytest.fixture
def sample_data() -> dict[str, list[int] | list[float]]:
    """Sample test-day milk recording data."""
    return {
        "dim": [10, 30, 60, 90, 120, 150, 200, 250, 305],
        "milkrecordings": [
            15.0,
            25.0,
            30.0,
            28.0,
            26.0,
            24.0,
            22.0,
            20.0,
            18.0,
        ],
    }


@pytest.fixture
def api() -> Generator[httpx.Client, None, None]:
    """HTTP client for testing the lactation-curves API.

    When API_BASE_URL is set, tests run against that live server.
    Otherwise, the FastAPI app is loaded in-process via ASGI transport.
    """
    base_url = os.getenv("API_BASE_URL", "")
    if base_url:
        with httpx.Client(base_url=base_url, timeout=30) as client:
            yield client
    else:
        from starlette.testclient import TestClient

        from main import app

        with TestClient(app, base_url="http://testserver") as client:
            yield client
