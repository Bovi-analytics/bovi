"""Shared fixtures for lactation-curves API tests.

Run tests:  uv run pytest
Or test against a deployed instance:
  API_BASE_URL=https://your-func.azurewebsites.net pytest
"""

import asyncio
import os
import sys
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest

_APP_DIR = Path(__file__).resolve().parents[1]
_APP_MODULES = ("main", "settings")


def _prepare_app_imports() -> None:
    """Ensure bare app imports resolve to this Function App in shared pytest runs."""
    app_path = str(_APP_DIR)
    if app_path in sys.path:
        sys.path.remove(app_path)
    sys.path.insert(0, app_path)

    for module_name in _APP_MODULES:
        loaded = sys.modules.get(module_name)
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file is not None and Path(loaded_file).resolve().parent != _APP_DIR:
            sys.modules.pop(module_name, None)


@pytest.fixture(autouse=True)
def app_import_context() -> None:
    _prepare_app_imports()


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


class ASGITestClient:
    """Small sync wrapper around httpx ASGITransport for local app tests."""

    def __init__(self, app) -> None:
        self._app = app

    def get(self, path: str, **kwargs) -> httpx.Response:
        return asyncio.run(self._request("GET", path, **kwargs))

    def post(self, path: str, **kwargs) -> httpx.Response:
        return asyncio.run(self._request("POST", path, **kwargs))

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=30,
        ) as client:
            return await client.request(method, path, **kwargs)


@pytest.fixture
def api() -> Generator[httpx.Client | ASGITestClient, None, None]:
    """HTTP client for testing the lactation-curves API.

    When API_BASE_URL is set, tests run against that live server.
    Otherwise, the FastAPI app is loaded in-process via ASGI transport.
    """
    base_url = os.getenv("API_BASE_URL", "")
    if base_url:
        with httpx.Client(base_url=base_url, timeout=30) as client:
            yield client
    else:
        _prepare_app_imports()
        from main import app

        yield ASGITestClient(app)
