"""Integration tests for benchmark routes."""

import asyncio
import json

import bovi_api.routes.benchmark as benchmark_routes
import pytest
from bovi_api.database import get_session
from bovi_api.models import Challenge, Submission, UploadAudit
from bovi_api.routes.benchmark import _dispatch_model
from bovi_api.settings import Settings
from bovi_api.upload_storage import UploadStorageError
from sqlmodel import select


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

    async def _audit() -> UploadAudit:
        async for session in override():
            result = await session.execute(select(UploadAudit))
            return result.scalars().one()
        raise AssertionError("session override did not yield")

    audit = asyncio.run(_audit())
    assert audit.action_type == "benchmark_submission_results"
    assert audit.status == "rejected"
    assert audit.user_id == 1
    assert audit.user_email == "user@example.test"
    assert audit.user_name == "Test User"
    assert audit.organization_id == 1
    assert audit.organization_name == "Test Organization"
    assert "Too many invalid rows" in (audit.error_detail or "")


def test_challenge_upload_stores_both_source_csv_audits(client):
    """Upload-backed challenges preserve both source CSVs in the audit log."""
    test_day = b"cow_id,parity,dim,milk_kg\ncow1,1,10,25.0\ncow1,1,40,30.0\ncow2,2,15,27.0\n"
    actual = b"cow_id,total_305_yield\ncow1,8000\ncow2,9000\n"

    response = client.post(
        "/benchmark/challenges/upload",
        data={"name": "Uploaded cohort"},
        files={
            "test_day_csv": ("test_day.csv", test_day, "text/csv"),
            "actual_yields_csv": ("actual.csv", actual, "text/csv"),
        },
    )

    assert response.status_code == 201
    challenge_id = response.json()["id"]
    override = client.app.dependency_overrides[get_session]

    async def _audits() -> list[UploadAudit]:
        async for session in override():
            result = await session.execute(select(UploadAudit).order_by(UploadAudit.action_type))
            return list(result.scalars().all())
        raise AssertionError("session override did not yield")

    audits = asyncio.run(_audits())
    assert {a.action_type for a in audits} == {
        "benchmark_challenge_actual_yields",
        "benchmark_challenge_test_day",
    }
    assert {a.status for a in audits} == {"accepted"}
    assert {a.challenge_id for a in audits} == {challenge_id}
    assert {a.user_email for a in audits} == {"user@example.test"}
    assert {a.organization_name for a in audits} == {"Test Organization"}


def test_submission_upload_stores_blob_audit_and_auth_identity(client, monkeypatch):
    """Accepted own-method CSV submissions are linked to a raw blob audit record."""
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> int:
        async for session in override():
            challenge = Challenge(
                dataset="icar",
                size="full",
                period="all",
                source="preset",
                name="seed",
                cow_metadata={
                    "cow1": {"parity": 1, "dim": [50], "milk_kg": [25.0]},
                    "cow2": {"parity": 2, "dim": [60], "milk_kg": [30.0]},
                },
                reference_yields=None,
                actual_yields={"cow1": 8000.0, "cow2": 9000.0},
                user_id=1,
                organization_id=1,
            )
            session.add(challenge)
            await session.commit()
            await session.refresh(challenge)
            assert challenge.id is not None
            return challenge.id
        raise AssertionError("session override did not yield")

    challenge_id = asyncio.run(_seed())

    async def _fake_dispatch(model, cow_metadata, settings):
        return {cid: 8100.0 for cid in cow_metadata}

    monkeypatch.setattr(benchmark_routes, "_dispatch_model", _fake_dispatch)

    resp = client.post(
        f"/benchmark/challenges/{challenge_id}/submissions/upload",
        data={
            "benchmark": "tim",
            "organization": "Bovi Labs",
            "country": "NL",
            "calculation_method": "spreadsheet",
        },
        files={
            "file": (
                "results.csv",
                b"cow_id,yield_305day\ncow1,8020\ncow2,9040\n",
                "text/csv",
            )
        },
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["user_id"] == 1
    assert body["organization_id"] == 1

    async def _audit() -> UploadAudit:
        async for session in override():
            result = await session.execute(select(UploadAudit))
            return result.scalars().one()
        raise AssertionError("session override did not yield")

    audit = asyncio.run(_audit())
    assert audit.status == "accepted"
    assert audit.action_type == "benchmark_submission_results"
    assert audit.original_filename == "results.csv"
    assert audit.challenge_id == challenge_id
    assert audit.submission_id == body["id"]
    assert audit.user_email == "user@example.test"
    assert audit.organization_name == "Test Organization"


def test_submission_upload_blocks_when_blob_storage_fails(client, monkeypatch):
    """No submission is saved when raw CSV storage cannot complete."""
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> int:
        async for session in override():
            challenge = Challenge(
                dataset="icar",
                size="full",
                period="all",
                source="preset",
                name="seed",
                cow_metadata={"cow1": {"parity": 1, "dim": [50], "milk_kg": [25.0]}},
                reference_yields=None,
                actual_yields={"cow1": 8000.0},
                user_id=1,
                organization_id=1,
            )
            session.add(challenge)
            await session.commit()
            await session.refresh(challenge)
            assert challenge.id is not None
            return challenge.id
        raise AssertionError("session override did not yield")

    challenge_id = asyncio.run(_seed())

    def _raise_storage_error(*args, **kwargs):
        raise UploadStorageError("storage unavailable")

    monkeypatch.setattr(benchmark_routes, "upload_csv_to_blob", _raise_storage_error)

    resp = client.post(
        f"/benchmark/challenges/{challenge_id}/submissions/upload",
        files={"file": ("results.csv", b"cow_id,yield_305day\ncow1,8020\n", "text/csv")},
    )

    assert resp.status_code == 503

    async def _counts() -> tuple[int, int]:
        async for session in override():
            submissions = await session.execute(select(Submission))
            audits = await session.execute(select(UploadAudit))
            return len(submissions.scalars().all()), len(audits.scalars().all())
        raise AssertionError("session override did not yield")

    assert asyncio.run(_counts()) == (0, 0)


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
