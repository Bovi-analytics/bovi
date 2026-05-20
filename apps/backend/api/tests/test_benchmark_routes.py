"""Integration tests for benchmark routes."""

import asyncio
import json

import bovi_api.routes.benchmark as benchmark_routes
import pytest
from bovi_api.database import get_session
from bovi_api.models import Challenge, Submission
from bovi_api.routes.benchmark import _dispatch_model
from bovi_api.routes.datasets import PresetCow, PresetDatasetResponse
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


def test_download_report_repairs_legacy_icar_actual_yields(client, monkeypatch):
    """Existing preset challenges with concatenated ALY are repaired before PDF export."""
    override = client.app.dependency_overrides[get_session]
    ids: dict[str, int] = {}

    async def _seed() -> None:
        async for session in override():
            challenge = Challenge(
                dataset="icar",
                size="full",
                period="all",
                source="preset",
                name="ICAR cohort",
                cow_metadata={"1483": {"parity": 2, "dim": [10, 40], "milk_kg": [30.0, 31.0]}},
                reference_yields=None,
                actual_yields={"1483": 148310573.1},
            )
            session.add(challenge)
            await session.commit()
            await session.refresh(challenge)
            assert challenge.id is not None
            submission = Submission(
                challenge_id=challenge.id,
                submission_type="bovi_model",
                model_type="wood",
                benchmark_model="tim",
                submitted_yields={"1483": 10500.0},
                bovi_yields={"1483": 10600.0},
                stats={
                    "version": 2,
                    "challenger_vs_aly": {
                        "overall": {"rmse": 148300073.1, "n": 1},
                        "by_parity": {},
                    },
                    "benchmark_vs_aly": {
                        "overall": {"rmse": 148299973.1, "n": 1},
                        "by_parity": {},
                    },
                    "challenger_vs_benchmark": {
                        "overall": {"rmse": 100.0, "n": 1},
                        "by_parity": {},
                    },
                    "failed_count": 0,
                },
                failed_cow_ids=[],
            )
            session.add(submission)
            await session.commit()
            await session.refresh(submission)
            assert submission.id is not None
            ids["submission"] = submission.id
            ids["challenge"] = challenge.id
            break

    asyncio.run(_seed())

    monkeypatch.setattr(
        benchmark_routes,
        "fetch_preset_cows",
        lambda *args, **kwargs: PresetDatasetResponse(
            dataset="icar",
            size="full",
            period="all",
            cow_count=1,
            cows=[
                PresetCow(
                    cow_id="1483",
                    display_name="Cow 1483 - parity 2",
                    parity=2,
                    dim=[10, 40],
                    milk_kg=[30.0, 31.0],
                )
            ],
            actual_yields={"1483": 10573.1},
        ),
    )

    captured: dict = {}

    def _fake_pdf(**kwargs):
        captured.update(kwargs)
        return b"%PDF-1.4\n"

    monkeypatch.setattr(benchmark_routes, "generate_report_pdf", _fake_pdf)

    response = client.get(f"/benchmark/submissions/{ids['submission']}/report")

    assert response.status_code == 200
    assert captured["actual_yields"] == {"1483": 10573.1}
    assert captured["stats"]["challenger_vs_aly"]["overall"]["rmse"] == pytest.approx(73.1)

    challenge = client.get(f"/benchmark/challenges/{ids['challenge']}").json()
    assert challenge["actual_yields"] == {"1483": 10573.1}
