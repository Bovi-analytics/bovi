"""Integration tests for benchmark routes."""

import asyncio
import gzip
import json

import bovi_api.routes.benchmark as benchmark_routes
import pytest
from bovi_api.database import get_session
from bovi_api.models import Challenge, StorageArtifact, Submission
from bovi_api.routes.benchmark import MilkBotRunOptions, _dispatch_model
from bovi_api.routes.datasets import PresetCow, PresetDatasetResponse
from bovi_api.settings import Settings
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


def test_create_challenge_from_saved_dataset(client):
    """POST /benchmark/challenges/saved-dataset creates an upload-backed challenge."""
    resp = client.post(
        "/benchmark/challenges/saved-dataset",
        json={
            "name": "Saved upload",
            "dataset_sources": [
                {
                    "role": "test_day_records",
                    "label": "Test-day records",
                    "filename": "test_day.csv",
                },
                {
                    "role": "actual_yields",
                    "label": "Ground-truth ALY",
                    "filename": "actual_yields.csv",
                },
            ],
            "cow_metadata": {
                "cow1": {"parity": 2, "herd_id": 123, "dim": [10, 40], "milk_kg": [25.0, 31.0]},
                "cow2": {"parity": 3, "herd_id": 123, "dim": [12, 42], "milk_kg": [28.0, 34.0]},
            },
            "actual_yields": {"cow1": 8500.0, "cow2": 9100.0},
        },
    )

    assert resp.status_code == 201
    data = resp.json()
    assert data["dataset"] == "saved_upload"
    assert data["source"] == "upload"
    assert data["name"] == "Saved upload"
    assert data["cow_count"] == 2
    assert data["dataset_sources"][0]["filename"] == "test_day.csv"
    assert data["dataset_stats"] == {
        "lactation_count": 2,
        "test_day_row_count": 4,
        "actual_yield_count": 2,
        "herd_count": 1,
    }

    detail = client.get(f"/benchmark/challenges/{data['id']}")
    assert detail.status_code == 200
    assert detail.json()["cow_metadata"]["cow1"]["milk_kg"] == [25.0, 31.0]


def test_create_upload_challenge_stores_raw_and_parsed_artifacts(client):
    """POST /benchmark/challenges/upload stores raw CSV and parsed JSON blobs."""
    test_day_csv = (
        b"TestId,parity,dim,milk_kg\n"
        b"cow1,2,10,25.0\n"
        b"cow1,2,40,31.0\n"
        b"cow2,3,12,28.0\n"
        b"cow2,3,42,34.0\n"
    )
    actual_yields_csv = b"TestId,LactationYield\ncow1,8500.0\ncow2,9100.0\n"

    resp = client.post(
        "/benchmark/challenges/upload",
        data={"name": "Uploaded"},
        files={
            "test_day_csv": ("farm_test_day.csv", test_day_csv, "text/csv"),
            "actual_yields_csv": ("farm_actual_yields.csv", actual_yields_csv, "text/csv"),
        },
    )

    assert resp.status_code == 201
    data = resp.json()
    assert data["row_count"] == 4
    assert data["cow_count"] == 2
    assert data["actual_yield_count"] == 2
    assert data["dataset_sources"] == [
        {
            "role": "test_day_records",
            "label": "Test-day records",
            "filename": "farm_test_day.csv",
        },
        {
            "role": "actual_yields",
            "label": "Ground-truth ALY",
            "filename": "farm_actual_yields.csv",
        },
    ]
    assert data["dataset_stats"]["lactation_count"] == 2
    assert data["dataset_stats"]["test_day_row_count"] == 4
    assert data["dataset_stats"]["actual_yield_count"] == 2

    blob_paths = set(client.app.state.blob_container_client.store)
    assert any(path.endswith("/raw/test_day.csv") for path in blob_paths)
    assert any(path.endswith("/raw/actual_yields.csv") for path in blob_paths)
    cow_blob_path = next(
        path for path in blob_paths if path.endswith("/parsed/cow_metadata.json.gz")
    )
    stored = client.app.state.blob_container_client.store[cow_blob_path]
    assert stored.content_type == "application/json"
    assert stored.content_encoding == "gzip"
    assert json.loads(gzip.decompress(stored.data))["cow1"]["dim"] == [10, 40]

    override = client.app.dependency_overrides[get_session]

    async def _artifact_count() -> int:
        async for session in override():
            result = await session.execute(select(StorageArtifact))
            return len(result.scalars().all())
        raise AssertionError("session override did not yield")

    assert asyncio.run(_artifact_count()) == 4


def test_create_preset_challenge_includes_icar_sources(client, monkeypatch):
    """Preset challenges expose the two ICAR source files in list responses."""
    monkeypatch.setattr(
        benchmark_routes,
        "fetch_preset_cows",
        lambda dataset, size, period, settings: PresetDatasetResponse(
            dataset=dataset,
            size=size,
            period=period,
            cow_count=1,
            cows=[
                PresetCow(
                    cow_id="cow1",
                    display_name="Cow 1",
                    parity=1,
                    dim=[10, 40],
                    milk_kg=[25.0, 31.0],
                )
            ],
            actual_yields={"cow1": 8500.0},
        ),
    )

    resp = client.post("/benchmark/challenges", json={"source": "preset", "preset": "icar"})

    assert resp.status_code == 201
    data = resp.json()
    assert [source["filename"] for source in data["dataset_sources"]] == [
        "TestDataSet.csv",
        "ActualMilkYields.csv",
    ]
    assert data["dataset_stats"]["lactation_count"] == 1
    assert data["dataset_stats"]["test_day_row_count"] == 2


def test_list_challenges_falls_back_dataset_metadata_for_legacy_rows(client):
    """Old rows without stored metadata still return usable list-view metadata."""
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> None:
        async for session in override():
            session.add(
                Challenge(
                    dataset="icar",
                    size="full",
                    period="all",
                    source="preset",
                    name="Legacy",
                    cow_metadata={
                        "cow1": {
                            "parity": 1,
                            "herd_id": None,
                            "dim": [10, 40],
                            "milk_kg": [25.0, 31.0],
                        }
                    },
                    reference_yields=None,
                    actual_yields={"cow1": 8500.0},
                )
            )
            await session.commit()
            break

    asyncio.run(_seed())

    resp = client.get("/benchmark/challenges")

    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["dataset_sources"][0]["filename"] == "TestDataSet.csv"
    assert data[0]["dataset_stats"]["test_day_row_count"] == 2


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
        b"TestId,LactationYield\n"
        + b"".join(f"cow{i},bad_value\n".encode() for i in range(9))
        + b"cow9,8000.0\n"
    )
    resp = client.post(
        "/benchmark/challenges/1/submissions/upload",
        files={"file": ("results.csv", csv_content, "text/csv")},
    )
    assert resp.status_code == 422


def test_submission_upload_stores_raw_and_generated_artifacts(client, monkeypatch):
    """Own-method submissions store raw CSV, parsed yields, and benchmark yields in blobs."""
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> None:
        async for session in override():
            session.add(
                Challenge(
                    dataset="upload",
                    size="custom",
                    period="custom",
                    source="upload",
                    name="seed",
                    cow_metadata={
                        "cow1": {"parity": 1, "dim": [50], "milk_kg": [25.0]},
                        "cow2": {"parity": 2, "dim": [60], "milk_kg": [30.0]},
                    },
                    reference_yields=None,
                    actual_yields={"cow1": 8000.0, "cow2": 9000.0},
                )
            )
            await session.commit()
            break

    async def _fake_dispatch(*_args, **_kwargs):
        return {"cow1": 7900.0, "cow2": 9100.0}

    asyncio.run(_seed())
    monkeypatch.setattr(benchmark_routes, "_dispatch_model", _fake_dispatch)

    resp = client.post(
        "/benchmark/challenges/1/submissions/upload",
        data={"benchmark": "tim", "calculation_method": "own method"},
        files={
            "file": (
                "results.csv",
                b"TestId,LactationYield\ncow1,8100\ncow2,9200\n",
                "text/csv",
            )
        },
    )

    assert resp.status_code == 201
    data = resp.json()
    assert data["row_count"] == 2
    assert data["submitted_yield_count"] == 2
    assert data["benchmark_yield_count"] == 2

    blob_paths = set(client.app.state.blob_container_client.store)
    assert any(path.endswith("/raw/results.csv") for path in blob_paths)
    assert any(path.endswith("/parsed/submitted_yields.json.gz") for path in blob_paths)
    assert any(path.endswith("/generated/bovi_yields.json.gz") for path in blob_paths)


def test_export_challenge_uses_test_id_and_omits_empty_herd_id(client):
    """Downloaded test data omits herd_id when all herd IDs are empty."""
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> None:
        async for session in override():
            session.add(
                Challenge(
                    dataset="icar",
                    size="full",
                    period="all",
                    source="preset",
                    name="Reference dataset",
                    cow_metadata={
                        "lact-1": {"parity": 1, "herd_id": None, "dim": [10], "milk_kg": [25.0]}
                    },
                    reference_yields=None,
                    actual_yields={"lact-1": 8000.0},
                )
            )
            await session.commit()
            break

    asyncio.run(_seed())

    resp = client.get("/benchmark/challenges/1/export")

    assert resp.status_code == 200
    assert resp.text.splitlines() == ["TestId,parity,dim,milk_kg", "lact-1,1,10,25.0"]


def test_export_challenge_keeps_herd_id_when_available(client):
    """Downloaded test data keeps herd_id for cohorts where it is populated."""
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> None:
        async for session in override():
            session.add(
                Challenge(
                    dataset="upload",
                    size="custom",
                    period="custom",
                    source="upload",
                    name="Uploaded",
                    cow_metadata={
                        "lact-1": {"parity": 1, "herd_id": 123, "dim": [10], "milk_kg": [25.0]}
                    },
                    reference_yields=None,
                    actual_yields={"lact-1": 8000.0},
                )
            )
            await session.commit()
            break

    asyncio.run(_seed())

    resp = client.get("/benchmark/challenges/1/export")

    assert resp.status_code == 200
    assert resp.text.splitlines() == ["TestId,herd_id,parity,dim,milk_kg", "lact-1,123,1,10,25.0"]


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

    async def post(self, url, content, headers, **kwargs):
        self.calls.append(
            {
                "url": url,
                "payload": json.loads(content),
                "headers": headers,
                **kwargs,
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


class _AutoencoderFakeClient:
    def __init__(self):
        self.calls = []

    async def post(self, url, content, headers, **kwargs):
        self.calls.append(
            {
                "url": url,
                "payload": json.loads(content),
                "headers": headers,
                **kwargs,
            }
        )
        return _FakeResponse(
            {
                "results": [
                    {"predictions": [1.0] * 304},
                    {"predictions": [2.0] * 304},
                ]
            }
        )


class _CharacteristicFakeClient:
    def __init__(self):
        self.calls = []

    async def post(self, url, content, headers, **kwargs):
        self.calls.append(
            {
                "url": url,
                "payload": json.loads(content),
                "headers": headers,
                **kwargs,
            }
        )
        return _FakeResponse(
            {
                "results": [
                    {"id": item["id"], "value": 9001.0 + index}
                    for index, item in enumerate(json.loads(content)["items"])
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


def test_dispatch_autoencoder_sends_parity_and_herd_id(monkeypatch):
    """Autoencoder benchmark dispatch preserves per-cow metadata in the batch payload."""
    fake_client = _AutoencoderFakeClient()
    monkeypatch.setattr(benchmark_routes, "_get_client", lambda: fake_client)

    cow_metadata = {
        "cow-1": {"dim": [10, 40], "milk_kg": [22.0, 31.0], "parity": 2, "herd_id": 2942694},
        "cow-2": {"dim": [12, 45], "milk_kg": [24.0, 33.0], "parity": 3, "herd_id": 991},
    }

    result = asyncio.run(
        _dispatch_model(
            "autoencoder",
            cow_metadata,
            Settings(lactation_autoencoder_url="https://autoencoder.example"),
        )
    )

    assert result == {"cow-1": 304.0, "cow-2": 608.0}
    assert fake_client.calls == [
        {
            "url": "https://autoencoder.example/predict/batch",
            "payload": {
                "items": [
                    {
                        "milk": [None] * 9 + [22.0] + [None] * 29 + [31.0] + [None] * 264,
                        "parity": 2,
                        "herd_id": 2942694,
                    },
                    {
                        "milk": [None] * 11 + [24.0] + [None] * 32 + [33.0] + [None] * 259,
                        "parity": 3,
                        "herd_id": 991,
                    },
                ],
                "imputation_method": "forward_fill",
            },
            "headers": {"Content-Type": "application/json"},
        }
    ]


def test_dispatch_milkbot_sends_bayesian_options_per_cow(monkeypatch):
    """MilkBot benchmark dispatch forwards Bayesian fitting options to /characteristic/batch."""
    fake_client = _CharacteristicFakeClient()
    monkeypatch.setattr(benchmark_routes, "_get_client", lambda: fake_client)

    cow_metadata = {
        "cow-1": {"dim": [10, 40], "milk_kg": [22.0, 31.0], "parity": 2},
        "cow-2": {"dim": [12, 45], "milk_kg": [24.0, 33.0], "parity": 4},
    }

    result = asyncio.run(
        _dispatch_model(
            "milkbot",
            cow_metadata,
            Settings(lactation_curves_url="https://curves.example"),
            MilkBotRunOptions(fitting="bayesian", breed="J", continent="CHEN"),
        )
    )

    assert result == {"cow-1": 9001.0, "cow-2": 9002.0}
    assert fake_client.calls == [
        {
            "url": "https://curves.example/characteristic/batch",
            "payload": {
                "items": [
                    {
                        "id": "cow-1",
                        "dim": [10, 40],
                        "milkrecordings": [22.0, 31.0],
                        "model": "milkbot",
                        "characteristic": "cumulative_milk_yield",
                        "parity": 2,
                        "lactation_length": 305,
                        "fitting": "bayesian",
                        "breed": "J",
                        "continent": "CHEN",
                    },
                    {
                        "id": "cow-2",
                        "dim": [12, 45],
                        "milkrecordings": [24.0, 33.0],
                        "model": "milkbot",
                        "characteristic": "cumulative_milk_yield",
                        "parity": 4,
                        "lactation_length": 305,
                        "fitting": "bayesian",
                        "breed": "J",
                        "continent": "CHEN",
                    },
                ]
            },
            "headers": {"Content-Type": "application/json"},
            "timeout": 300.0,
        },
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
