"""Integration tests for benchmark routes."""

import asyncio
import json

import bovi_api.routes.benchmark as benchmark_routes
import pytest
from bovi_api.database import get_session
from bovi_api.models import Challenge
from bovi_api.routes.benchmark import _dispatch_model
from bovi_api.settings import Settings


def test_list_challenges_empty(client):
    """GET /benchmark/challenges returns empty list when no challenges exist."""
    response = client.get("/benchmark/challenges")
    assert response.status_code == 200
    assert response.json() == []


def test_get_submission_not_found(client):
    """GET /benchmark/submissions/999 → 404."""
    resp = client.get("/benchmark/submissions/999")
    assert resp.status_code == 404


def test_list_submissions_empty(client):
    """GET /benchmark/submissions → []."""
    resp = client.get("/benchmark/submissions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_pad_b_upload_rejects_high_failure_rate(client):
    """POST upload with >20% bad rows → 422.

    Seeds a challenge directly via the override_get_session sessionmaker so that
    the upload route gets past the "challenge not found" check and reaches the
    threshold logic.
    """
    # Reuse the same session machinery the override does - find the engine
    # by inspecting the dependency override on the test app.
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> None:
        # The override is an async generator; pull its session factory by calling it
        async for session in override():
            session.add(
                Challenge(
                    dataset="icar",
                    size="full",
                    period="all",
                    source="preset",
                    name="seed",
                    cow_metadata={
                        f"cow{i}": {"parity": 1, "dim": [50], "milk_kg": [25.0]} for i in range(10)
                    },
                    reference_yields=None,
                    actual_yields={f"cow{i}": 8000.0 for i in range(10)},
                    user_id=1,
                    organization_id=1,
                )
            )
            await session.commit()
            break

    asyncio.run(_seed())

    csv_content = (
        b"cow_id,yield_305day\n"
        + b"".join(f"cow{i},bad_value\n".encode() for i in range(9))
        + b"cow9,8000.0\n"
    )
    resp = client.post(
        "/benchmark/challenges/1/submissions/upload",
        files={"file": ("results.csv", csv_content, "text/csv")},
    )
    assert resp.status_code == 422


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self):
        self.calls = []

    async def post(self, url, content, headers):
        self.calls.append(
            {
                "url": url,
                "payload": json.loads(content),
                "headers": headers,
            }
        )
        return _FakeResponse(
            {
                "results": [
                    {"test_id": "cow-1", "total_305_yield": 8123.4},
                    {"test_id": "cow-2", "total_305_yield": 9345.6},
                ]
            }
        )


@pytest.mark.parametrize(
    ("model", "expected_path"),
    [
        ("tim", "/test-interval"),
        ("islc", "/islc"),
        ("best_predict", "/best-predict"),
    ],
)
def test_dispatch_yield_estimators_call_matching_lactation_curve_endpoint(
    monkeypatch,
    model,
    expected_path,
):
    """Benchmark yield estimators are dispatched to their dedicated upstream endpoints."""
    fake_client = _FakeClient()
    monkeypatch.setattr(benchmark_routes, "_get_client", lambda: fake_client)

    cow_metadata = {
        "cow-1": {"dim": [10, 40], "milk_kg": [22.0, 31.0], "parity": 1},
        "cow-2": {"dim": [12, 45], "milk_kg": [24.0, 33.0], "parity": 2},
    }

    result = asyncio.run(
        _dispatch_model(
            model,
            cow_metadata,
            Settings(lactation_curves_url="https://curves.example"),
        )
    )

    assert result == {"cow-1": 8123.4, "cow-2": 9345.6}
    assert fake_client.calls == [
        {
            "url": f"https://curves.example{expected_path}",
            "payload": {
                "dim": [10, 40, 12, 45],
                "milkrecordings": [22.0, 31.0, 24.0, 33.0],
                "test_ids": ["cow-1", "cow-1", "cow-2", "cow-2"],
            },
            "headers": {"Content-Type": "application/json"},
        }
    ]
