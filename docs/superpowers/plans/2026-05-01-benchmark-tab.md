# Benchmark Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/benchmark` tab to the bovi dashboard that implements the ICAR accreditation workflow — create test challenges from preset cow datasets, submit 305-day yield calculations (via bovi models or own CSV), compare against ICAR reference values, and download a PDF report.

**Architecture:** New FastAPI router (`benchmark.py`) backed by two new DB tables (`challenges`, `submissions`). Frontend is a new Next.js route group with React Query hooks. The benchmark route reuses the existing `POST /curves/test-interval` proxy and extracts a shared `fetch_preset_cows()` utility from `datasets.py`. Auth is deferred — schema has nullable `user_id` fields.

**Tech Stack:** Python 3.12 / FastAPI / SQLModel / SQLite with aiosqlite / scipy / scikit-learn / fpdf2 — Next.js 14 / TypeScript / Mantine / TanStack React Query / Zod

---

## File Map

**Backend — new files:**
- `apps/backend/api/src/bovi_api/benchmark_ingestion.py` — `parse_submission_csv()`
- `apps/backend/api/src/bovi_api/benchmark_stats.py` — `calculate_comparison_stats()`
- `apps/backend/api/src/bovi_api/benchmark_pdf.py` — `generate_report_pdf()`
- `apps/backend/api/src/bovi_api/routes/benchmark.py` — all benchmark endpoints
- `apps/backend/api/alembic/versions/0002_add_benchmark_tables.py` — migration
- `apps/backend/api/tests/test_benchmark_ingestion.py`
- `apps/backend/api/tests/test_benchmark_stats.py`
- `apps/backend/api/tests/test_benchmark_routes.py`

**Backend — modified files:**
- `apps/backend/api/pyproject.toml` — add scipy, scikit-learn, fpdf2
- `apps/backend/api/src/bovi_api/models.py` — add Challenge + Submission
- `apps/backend/api/src/bovi_api/routes/datasets.py` — extract `fetch_preset_cows()`
- `apps/backend/api/src/bovi_api/app.py` — register benchmark router

**Frontend — new files:**
- `apps/frontend/dashboard/src/app/(dashboard)/benchmark/page.tsx`
- `apps/frontend/dashboard/src/app/(dashboard)/benchmark/new/page.tsx`
- `apps/frontend/dashboard/src/app/(dashboard)/benchmark/[id]/page.tsx`
- `apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/challenge-card.tsx`
- `apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/submission-form.tsx`
- `apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/submission-form-bovi.tsx`
- `apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/submission-form-upload.tsx`
- `apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/comparison-results.tsx`
- `apps/frontend/dashboard/src/app/(dashboard)/benchmark/hooks/use-challenges.ts`
- `apps/frontend/dashboard/src/app/(dashboard)/benchmark/hooks/use-submissions.ts`

**Frontend — modified files:**
- `apps/frontend/dashboard/src/types/api.ts`
- `apps/frontend/dashboard/src/lib/api-client.ts`
- `apps/frontend/dashboard/src/components/dashboard/navigation.ts`

---

## Task 1: Add backend dependencies

**Files:**
- Modify: `apps/backend/api/pyproject.toml`

- [ ] **Step 1: Add scipy, scikit-learn, fpdf2 to pyproject.toml**

```toml
# In [project] dependencies list, add:
"scipy>=1.13",
"scikit-learn>=1.5",
"fpdf2>=2.7",
```

- [ ] **Step 2: Sync dependencies**

```bash
cd apps/backend/api && uv sync
```

Expected: packages installed, no errors.

- [ ] **Step 3: Verify imports work**

```bash
cd apps/backend/api && uv run python -c "import scipy.stats; import sklearn.metrics; from fpdf import FPDF; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 4: Commit**

```bash
git add apps/backend/api/pyproject.toml apps/backend/api/uv.lock
git commit -m "feat(benchmark): add scipy, scikit-learn, fpdf2 dependencies"
```

---

## Task 2: parse_submission_csv()

Parses a Pad B CSV upload (`cow_id,yield_305day`) into a dict. Distinct from `parse_csv()` in `herd_stats_ingestion.py` which aggregates to herd level.

**Files:**
- Create: `apps/backend/api/src/bovi_api/benchmark_ingestion.py`
- Create: `apps/backend/api/tests/test_benchmark_ingestion.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/backend/api/tests/test_benchmark_ingestion.py
"""Tests for benchmark CSV ingestion."""

import pytest
from bovi_api.benchmark_ingestion import parse_submission_csv


def test_parse_valid_csv():
    csv_bytes = b"cow_id,yield_305day\ncow1,8500.0\ncow2,9200.5\n"
    result = parse_submission_csv(csv_bytes)
    assert result == {"cow1": 8500.0, "cow2": 9200.5}


def test_parse_skips_invalid_rows_and_returns_failed():
    csv_bytes = b"cow_id,yield_305day\ncow1,8500.0\ncow2,not_a_number\ncow3,7000.0\n"
    result, failed = parse_submission_csv(csv_bytes, return_failed=True)
    assert result == {"cow1": 8500.0, "cow3": 7000.0}
    assert failed == ["cow2"]


def test_parse_missing_required_column_raises():
    csv_bytes = b"cow_id,wrong_column\ncow1,8500.0\n"
    with pytest.raises(ValueError, match="yield_305day"):
        parse_submission_csv(csv_bytes)


def test_parse_empty_file_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_submission_csv(b"")


def test_parse_negative_yield_raises():
    csv_bytes = b"cow_id,yield_305day\ncow1,-100.0\n"
    with pytest.raises(ValueError, match="negative"):
        parse_submission_csv(csv_bytes)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd apps/backend/api && uv run pytest tests/test_benchmark_ingestion.py -v
```

Expected: ImportError or AttributeError (module does not exist yet).

- [ ] **Step 3: Implement parse_submission_csv()**

```python
# apps/backend/api/src/bovi_api/benchmark_ingestion.py
"""CSV ingestion for Pad B benchmark submissions."""

from __future__ import annotations

import csv
import io


def parse_submission_csv(
    content: bytes,
    return_failed: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[str]]:
    """Parse a Pad B CSV upload into {cow_id: yield_305day}.

    Args:
        content: Raw CSV bytes. Required columns: cow_id, yield_305day.
        return_failed: If True, return (yields, failed_cow_ids) tuple.

    Returns:
        dict[str, float] or (dict[str, float], list[str]) if return_failed=True.

    Raises:
        ValueError: If file is empty, missing required columns, or all rows invalid.

    """
    if not content.strip():
        raise ValueError("CSV file is empty.")

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV file is not valid UTF-8.") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV file has no headers.")

    normalised = {h.strip().lower(): h for h in reader.fieldnames}
    if "cow_id" not in normalised or "yield_305day" not in normalised:
        missing = [c for c in ("cow_id", "yield_305day") if c not in normalised]
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    cow_col = normalised["cow_id"]
    yield_col = normalised["yield_305day"]

    results: dict[str, float] = {}
    failed: list[str] = []

    for row in reader:
        cow_id = row[cow_col].strip()
        try:
            value = float(row[yield_col])
        except (ValueError, KeyError):
            failed.append(cow_id)
            continue
        if value < 0:
            raise ValueError(f"Negative yield for cow '{cow_id}': {value}")
        results[cow_id] = value

    if return_failed:
        return results, failed
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd apps/backend/api && uv run pytest tests/test_benchmark_ingestion.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/backend/api/src/bovi_api/benchmark_ingestion.py \
        apps/backend/api/tests/test_benchmark_ingestion.py
git commit -m "feat(benchmark): add parse_submission_csv for Pad B CSV uploads"
```

---

## Task 3: calculate_comparison_stats()

Computes Pearson, RMSE, MAE, MAPE overall and split by parity (1, 2, 3+).

**Files:**
- Create: `apps/backend/api/src/bovi_api/benchmark_stats.py`
- Create: `apps/backend/api/tests/test_benchmark_stats.py`

- [ ] **Step 1: Write the failing tests**

```python
# apps/backend/api/tests/test_benchmark_stats.py
"""Tests for benchmark statistics calculation."""

import pytest
from bovi_api.benchmark_stats import calculate_comparison_stats


def _make_identical(n: int = 10) -> tuple[dict, dict, dict]:
    """Perfect submission: submitted == reference."""
    ids = [str(i) for i in range(n)]
    yields = {cid: 8000.0 + i * 100 for i, cid in enumerate(ids)}
    parities = {cid: (i % 3) + 1 for i, cid in enumerate(ids)}
    return yields, dict(yields), parities


def test_identical_submission_has_pearson_one():
    submitted, reference, parities = _make_identical(20)
    stats = calculate_comparison_stats(submitted, reference, parities)
    assert stats["overall"]["pearson"] == pytest.approx(1.0, abs=1e-6)


def test_identical_submission_has_zero_rmse():
    submitted, reference, parities = _make_identical(20)
    stats = calculate_comparison_stats(submitted, reference, parities)
    assert stats["overall"]["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_stats_contain_required_keys():
    submitted, reference, parities = _make_identical(20)
    stats = calculate_comparison_stats(submitted, reference, parities)
    assert "overall" in stats
    assert "by_parity" in stats
    assert "failed_count" in stats
    for key in ("pearson", "rmse", "mae", "mape", "n"):
        assert key in stats["overall"]


def test_parity_grouping_combines_3plus():
    ids = ["a", "b", "c", "d"]
    yields = {cid: 8000.0 for cid in ids}
    parities = {"a": 1, "b": 2, "c": 3, "d": 4}
    stats = calculate_comparison_stats(yields, dict(yields), parities)
    assert "3+" in stats["by_parity"]
    assert "3" not in stats["by_parity"]
    assert "4" not in stats["by_parity"]
    assert stats["by_parity"]["3+"]["n"] == 2


def test_missing_reference_cow_counted_as_failed():
    submitted = {"cow1": 8000.0, "cow2": 9000.0}
    reference = {"cow1": 8000.0}  # cow2 missing
    parities = {"cow1": 1, "cow2": 1}
    stats = calculate_comparison_stats(submitted, reference, parities)
    assert stats["failed_count"] == 1
    assert stats["overall"]["n"] == 1


def test_n_reflects_matched_cows():
    submitted, reference, parities = _make_identical(15)
    stats = calculate_comparison_stats(submitted, reference, parities)
    assert stats["overall"]["n"] == 15
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd apps/backend/api && uv run pytest tests/test_benchmark_stats.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement calculate_comparison_stats()**

```python
# apps/backend/api/src/bovi_api/benchmark_stats.py
"""Statistics for comparing submitted 305-day yields against reference values."""

from __future__ import annotations

import math

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_comparison_stats(
    submitted: dict[str, float],
    reference: dict[str, float],
    parities: dict[str, int],
) -> dict:
    """Compute Pearson, RMSE, MAE, MAPE overall and per parity group (1, 2, 3+).

    Args:
        submitted: {cow_id: yield_305day} from the submission.
        reference: {cow_id: yield_305day} ICAR TIM reference values.
        parities: {cow_id: parity_int} from challenge.cow_metadata.

    Returns:
        dict with keys "overall", "by_parity", "failed_count".

    """
    common = [cid for cid in submitted if cid in reference]
    failed_count = len(submitted) - len(common)

    def _stats_for(pairs: list[tuple[float, float]]) -> dict:
        if not pairs:
            return {"pearson": None, "rmse": None, "mae": None, "mape": None, "n": 0}
        sub = [p[0] for p in pairs]
        ref = [p[1] for p in pairs]
        corr, _ = pearsonr(sub, ref)
        rmse = math.sqrt(mean_squared_error(ref, sub))
        mae = mean_absolute_error(ref, sub)
        mape = sum(abs((r - s) / r) for s, r in zip(sub, ref) if r != 0) / len(pairs) * 100
        return {"pearson": round(corr, 6), "rmse": round(rmse, 3),
                "mae": round(mae, 3), "mape": round(mape, 3), "n": len(pairs)}

    all_pairs = [(submitted[cid], reference[cid]) for cid in common]

    parity_groups: dict[str, list[tuple[float, float]]] = {}
    for cid in common:
        p = parities.get(cid, 1)
        key = str(p) if p <= 2 else "3+"
        parity_groups.setdefault(key, []).append((submitted[cid], reference[cid]))

    return {
        "overall": _stats_for(all_pairs),
        "by_parity": {k: _stats_for(v) for k, v in sorted(parity_groups.items())},
        "failed_count": failed_count,
    }
```

- [ ] **Step 4: Run tests**

```bash
cd apps/backend/api && uv run pytest tests/test_benchmark_stats.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/backend/api/src/bovi_api/benchmark_stats.py \
        apps/backend/api/tests/test_benchmark_stats.py
git commit -m "feat(benchmark): add calculate_comparison_stats with parity breakdown"
```

---

## Task 4: DB models and migration

**Files:**
- Modify: `apps/backend/api/src/bovi_api/models.py`
- Create: `apps/backend/api/alembic/versions/0002_add_benchmark_tables.py`

- [ ] **Step 1: Add Challenge and Submission to models.py**

Add after the existing `HerdProfileRead` class:

```python
# At the top of models.py, add to imports:
from sqlalchemy import JSON

# --- Benchmark models ---

class ChallengeBase(SQLModel):
    """Shared fields for benchmark challenges."""
    dataset: str = Field(description="'aurora' or 'sunnyside'")
    size: str = Field(description="'small' or 'medium'")
    period: str = Field(description="'recent', 'old', or 'mixed'")
    user_id: str | None = Field(default=None, description="Auth-ready; nullable until auth is added")


class Challenge(ChallengeBase, table=True):
    """A benchmark challenge: a sampled set of cows with pre-computed reference yields."""

    __tablename__ = "challenges"

    id: int | None = Field(default=None, primary_key=True)
    cow_metadata: dict = Field(
        sa_column=Column(JSON),
        description="{cow_id: {parity, dim[], milk_kg[]}} — test-day records per cow",
    )
    reference_yields: dict = Field(
        sa_column=Column(JSON),
        description="{cow_id: float} — TIM-calculated 305-day reference yields",
    )
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=sa_func.now()),
    )


class ChallengeRead(ChallengeBase):
    """API response for a challenge (excludes large internal blobs for list views)."""
    id: int
    created_at: datetime | None


class ChallengeDetail(ChallengeRead):
    """Full challenge response including cow data (used for export and submission)."""
    cow_metadata: dict
    reference_yields: dict


class SubmissionBase(SQLModel):
    """Shared fields for benchmark submissions."""
    submission_type: str = Field(description="'bovi_model' or 'own_method'")
    model_type: str | None = Field(default=None, description="e.g. 'tim', 'wood'")
    organization: str | None = Field(default=None)
    country: str | None = Field(default=None)
    calculation_method: str | None = Field(default=None)
    notes: str | None = Field(default=None)
    user_id: str | None = Field(default=None)


class Submission(SubmissionBase, table=True):
    """A user's submission for a benchmark challenge."""

    __tablename__ = "submissions"

    id: int | None = Field(default=None, primary_key=True)
    challenge_id: int = Field(foreign_key="challenges.id")
    submitted_yields: dict = Field(
        sa_column=Column(JSON),
        description="{cow_id: float} — user-submitted or bovi-calculated yields",
    )
    bovi_yields: dict = Field(
        sa_column=Column(JSON),
        description="{cow_id: float} — TIM yields for report flavors 2 and 3",
    )
    stats: dict = Field(
        sa_column=Column(JSON),
        description="Comparison statistics (Pearson, RMSE, MAE, MAPE per parity)",
    )
    failed_cow_ids: list = Field(
        sa_column=Column(JSON),
        default_factory=list,
        description="Cow IDs excluded from stats due to parse/compute failure",
    )
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=sa_func.now()),
    )


class SubmissionRead(SubmissionBase):
    """API response for a submission."""
    id: int
    challenge_id: int
    stats: dict
    failed_cow_ids: list
    created_at: datetime | None
```

- [ ] **Step 2: Create Alembic migration**

```python
# apps/backend/api/alembic/versions/0002_add_benchmark_tables.py
"""add challenges and submissions tables

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-01
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "challenges",
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("size", sa.String(), nullable=False),
        sa.Column("period", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("cow_metadata", sa.JSON(), nullable=False),
        sa.Column("reference_yields", sa.JSON(), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "submissions",
        sa.Column("submission_type", sa.String(), nullable=False),
        sa.Column("model_type", sa.String(), nullable=True),
        sa.Column("organization", sa.String(), nullable=True),
        sa.Column("country", sa.String(), nullable=True),
        sa.Column("calculation_method", sa.String(), nullable=True),
        sa.Column("notes", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("challenge_id", sa.Integer(), nullable=False),
        sa.Column("submitted_yields", sa.JSON(), nullable=False),
        sa.Column("bovi_yields", sa.JSON(), nullable=False),
        sa.Column("stats", sa.JSON(), nullable=False),
        sa.Column("failed_cow_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["challenge_id"], ["challenges.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("submissions")
    op.drop_table("challenges")
```

- [ ] **Step 3: Verify migration runs**

```bash
cd apps/backend/api && uv run alembic upgrade head
```

Expected: "Running upgrade 0001 -> 0002".

- [ ] **Step 4: Commit**

```bash
git add apps/backend/api/src/bovi_api/models.py \
        apps/backend/api/alembic/versions/0002_add_benchmark_tables.py
git commit -m "feat(benchmark): add Challenge and Submission DB models with migration"
```

---

## Task 5: Extract fetch_preset_cows() from datasets.py

The benchmark route needs to fetch preset cow data without making an internal HTTP call to itself. Extract the blob-fetch logic into a reusable async-compatible function.

**Files:**
- Modify: `apps/backend/api/src/bovi_api/routes/datasets.py`

- [ ] **Step 1: Extract fetch_preset_cows() utility**

Add this function to `datasets.py`, above the route handlers. The existing `get_preset()` route will be refactored to call it.

```python
# Add to datasets.py after the _MANIFEST definition:

import json as _json  # move top-level import to top of file

def fetch_preset_cows(
    dataset: DatasetKey,
    size: SizeKey,
    period: PeriodKey,
    settings: Settings,
) -> PresetDatasetResponse:
    """Fetch cow data from Azure Blob — shared by routes and benchmark logic."""
    blob_path = f"{_BLOB_PREFIX}/{dataset}/{size}_{period}.json"
    client = _get_blob_client(settings)
    try:
        data = client.get_blob_client(_CONTAINER, blob_path).download_blob().readall()
    except ResourceNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Preset dataset not found: {blob_path}.",
        )
    payload = _json.loads(data)
    return PresetDatasetResponse(**payload)
```

Refactor the existing route to call it:

```python
@router.get("/presets/{dataset}/{size}/{period}", response_model=PresetDatasetResponse)
def get_preset(
    dataset: DatasetKey,
    size: SizeKey,
    period: PeriodKey,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PresetDatasetResponse:
    """Fetch a pre-generated cow-dataset JSON blob from Azure Blob Storage."""
    return fetch_preset_cows(dataset, size, period, settings)
```

Also remove the `import json` that was inline inside the old `get_preset()` body and add it at the top of the file.

- [ ] **Step 2: Run existing tests to verify no regression**

```bash
cd apps/backend/api && uv run pytest tests/ -v -k "not benchmark"
```

Expected: all existing tests PASS.

- [ ] **Step 3: Commit**

```bash
git add apps/backend/api/src/bovi_api/routes/datasets.py
git commit -m "refactor(datasets): extract fetch_preset_cows() for reuse by benchmark route"
```

---

## Task 6: Benchmark challenge endpoints (POST + GET list + GET detail + GET export)

**Files:**
- Create: `apps/backend/api/src/bovi_api/routes/benchmark.py`
- Create: `apps/backend/api/tests/test_benchmark_routes.py`

Before implementing: check the response format of `POST /curves/test-interval` when called with multiple `test_ids`. Run this command against a local instance, or read `apps/backend/models/lactation-curves/main.py` to find the `TestIntervalResponse` schema. The response format determines how `reference_yields` is built. Based on the existing TypeScript schema:
```typescript
export const TestIntervalResponseSchema = z.object({ ... })
```
The endpoint returns a list of `{test_id, yield_305day}` pairs. Adjust the parsing in the implementation below if the actual format differs.

- [ ] **Step 1: Write failing tests for challenge endpoints**

```python
# apps/backend/api/tests/test_benchmark_routes.py
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
```

Note: Full route integration tests require mocking the Azure Blob and TIM proxy calls. The tests above are a scaffold. Add mocks using `pytest-mock` or `unittest.mock.patch` as needed. The key contracts to verify:
- `GET /benchmark/challenges` → `[]` when empty
- `GET /benchmark/challenges/{id}` → 404 for unknown ID
- `GET /benchmark/challenges/{id}/export` → `text/csv` response

- [ ] **Step 2: Run failing tests**

```bash
cd apps/backend/api && uv run pytest tests/test_benchmark_routes.py -v
```

Expected: ImportError or 404 (router not registered yet).

- [ ] **Step 3: Create benchmark.py with challenge endpoints**

```python
# apps/backend/api/src/bovi_api/routes/benchmark.py
"""Benchmark endpoints — ICAR accreditation workflow."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from bovi_api.benchmark_ingestion import parse_submission_csv
from bovi_api.benchmark_stats import calculate_comparison_stats
from bovi_api.database import get_session
from bovi_api.models import (
    Challenge,
    ChallengeDetail,
    ChallengeRead,
    Submission,
    SubmissionRead,
)
from bovi_api.routes.datasets import DatasetKey, PeriodKey, SizeKey, fetch_preset_cows
from bovi_api.routes.proxy import _get_client
from bovi_api.settings import Settings, get_settings

logger = logging.getLogger("bovi_api.benchmark")
router = APIRouter(prefix="/benchmark", tags=["benchmark"])

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_FAILURE_THRESHOLD = 0.20  # >20% failed cows → reject submission


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _call_tim(
    cow_metadata: dict[str, dict],
    settings: Settings,
) -> dict[str, float]:
    """Call POST /curves/test-interval for all cows in cow_metadata.

    Args:
        cow_metadata: {cow_id: {parity, dim: list[int], milk_kg: list[float]}}
        settings: App settings (for lactation_curves_url).

    Returns:
        {cow_id: yield_305day}

    Raises:
        HTTPException 502 if upstream unreachable.

    """
    # Build flat arrays with test_ids to identify each measurement
    dim_all: list[int] = []
    milk_all: list[float] = []
    test_ids_all: list[str] = []

    for cow_id, meta in cow_metadata.items():
        for d, m in zip(meta["dim"], meta["milk_kg"]):
            dim_all.append(d)
            milk_all.append(m)
            test_ids_all.append(cow_id)

    payload = {"dim": dim_all, "milkrecordings": milk_all, "test_ids": test_ids_all}

    client = _get_client()
    try:
        resp = await client.post(
            f"{settings.lactation_curves_url}/test-interval",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
    except httpx.RequestError as exc:
        logger.exception("TIM proxy error: %s", exc)
        raise HTTPException(status_code=502, detail="Upstream lactation-curves service unavailable.")

    data = resp.json()
    # The Function App returns [{test_id, yield_305day}, ...] or {test_id: yield}
    # Normalise both formats to {cow_id: float}
    if isinstance(data, list):
        return {item["test_id"]: item["yield_305day"] for item in data}
    # dict format
    return {str(k): float(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Challenge endpoints
# ---------------------------------------------------------------------------


class ChallengeCreateBody(BaseModel):
    dataset: DatasetKey
    size: Literal["small", "medium"]  # "large" is out of scope (timeout risk)
    period: PeriodKey


@router.post("/challenges", response_model=ChallengeRead, status_code=201)
async def create_challenge(
    body: ChallengeCreateBody,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Challenge:
    """Create a challenge by sampling cows from a preset dataset and computing reference yields."""
    # fetch_preset_cows is synchronous (Azure Blob SDK); run in executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    preset = await loop.run_in_executor(
        None, fetch_preset_cows, body.dataset, body.size, body.period, settings
    )

    # Build cow_metadata from PresetCow objects
    cow_metadata: dict[str, dict] = {
        cow.cow_id: {
            "parity": cow.parity,
            "dim": cow.dim,
            "milk_kg": cow.milk_kg,
        }
        for cow in preset.cows
    }

    # Compute reference yields via TIM
    reference_yields = await _call_tim(cow_metadata, settings)

    challenge = Challenge(
        dataset=body.dataset,
        size=body.size,
        period=body.period,
        cow_metadata=cow_metadata,
        reference_yields=reference_yields,
    )
    session.add(challenge)
    await session.commit()
    await session.refresh(challenge)
    return challenge


@router.get("/challenges", response_model=list[ChallengeRead])
async def list_challenges(
    session: AsyncSession = Depends(get_session),
) -> list[Challenge]:
    """List all challenges, newest first."""
    result = await session.execute(
        select(Challenge).order_by(Challenge.created_at.desc()).limit(100)
    )
    return list(result.scalars().all())


@router.get("/challenges/{challenge_id}", response_model=ChallengeDetail)
async def get_challenge(
    challenge_id: int,
    session: AsyncSession = Depends(get_session),
) -> Challenge:
    """Get a single challenge with full cow metadata."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    return challenge


@router.get("/challenges/{challenge_id}/export")
async def export_challenge(
    challenge_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Download a CSV of the challenge's test-day cow data for Pad B users."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["cow_id", "parity", "dim", "milk_kg"])

    for cow_id, meta in challenge.cow_metadata.items():
        parity = meta.get("parity", "")
        for d, m in zip(meta["dim"], meta["milk_kg"]):
            writer.writerow([cow_id, parity, d, m])

    content = output.getvalue().encode("utf-8")
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=challenge_{challenge_id}.csv"},
    )
```

- [ ] **Step 4: Register the router in app.py**

In `apps/backend/api/src/bovi_api/app.py`, add:

```python
from bovi_api.routes import benchmark  # add to existing imports

# In create_app(), after the other include_router calls:
app.include_router(benchmark.router)
```

- [ ] **Step 5: Run tests**

```bash
cd apps/backend/api && uv run pytest tests/test_benchmark_routes.py -v
```

Expected: PASS (or clearer failures showing routes are reachable now).

- [ ] **Step 6: Commit**

```bash
git add apps/backend/api/src/bovi_api/routes/benchmark.py \
        apps/backend/api/src/bovi_api/app.py \
        apps/backend/api/tests/test_benchmark_routes.py
git commit -m "feat(benchmark): add challenge endpoints (create, list, get, export)"
```

---

## Task 7: Submission endpoints (Pad A + Pad B) and GET list/detail

**Files:**
- Modify: `apps/backend/api/src/bovi_api/routes/benchmark.py`
- Modify: `apps/backend/api/tests/test_benchmark_routes.py`

- [ ] **Step 1: Write failing tests for submission endpoints**

Add to `test_benchmark_routes.py`:

```python
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
```

- [ ] **Step 2: Add submission endpoints to benchmark.py**

Append to the existing `benchmark.py` after the challenge endpoints:

```python
# ---------------------------------------------------------------------------
# Submission endpoints
# ---------------------------------------------------------------------------


class SubmissionBodviModel(BaseModel):
    submission_type: str = "bovi_model"
    model_type: str = "tim"
    organization: str | None = None
    country: str | None = None
    notes: str | None = None


@router.post("/challenges/{challenge_id}/submissions", response_model=SubmissionRead, status_code=201)
async def create_submission_bovi(
    challenge_id: int,
    body: SubmissionBodviModel,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Submission:
    """Pad A: compute 305-day yields for all challenge cows using a bovi model, then submit."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")

    if body.model_type == "tim":
        bovi_yields = await _call_tim(challenge.cow_metadata, settings)
    else:
        raise HTTPException(
            status_code=422,
            detail=f"model_type '{body.model_type}' not supported yet. Use 'tim'.",
        )

    parities = {cid: meta.get("parity", 1) or 1 for cid, meta in challenge.cow_metadata.items()}
    stats = calculate_comparison_stats(bovi_yields, challenge.reference_yields, parities)

    submission = Submission(
        challenge_id=challenge_id,
        submission_type="bovi_model",
        model_type=body.model_type,
        organization=body.organization,
        country=body.country,
        submitted_yields=bovi_yields,
        bovi_yields=bovi_yields,
        stats=stats,
        failed_cow_ids=[],
    )
    session.add(submission)
    await session.commit()
    await session.refresh(submission)
    return submission


@router.post(
    "/challenges/{challenge_id}/submissions/upload",
    response_model=SubmissionRead,
    status_code=201,
)
async def create_submission_upload(
    challenge_id: int,
    file: UploadFile = File(...),
    organization: str | None = Form(default=None),
    country: str | None = Form(default=None),
    calculation_method: str | None = Form(default=None),
    notes: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Submission:
    """Pad B: upload a CSV of own-method yields, compare against reference."""
    challenge = await session.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")

    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")

    try:
        submitted_yields, failed_ids = parse_submission_csv(content, return_failed=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    total = len(submitted_yields) + len(failed_ids)
    if total > 0 and len(failed_ids) / total > _FAILURE_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail=f"Too many invalid rows: {len(failed_ids)}/{total} failed (>{int(_FAILURE_THRESHOLD*100)}% threshold).",
        )

    # Auto-compute bovi TIM yields for report flavors 2 and 3
    bovi_yields = await _call_tim(challenge.cow_metadata, settings)

    parities = {cid: meta.get("parity", 1) or 1 for cid, meta in challenge.cow_metadata.items()}
    stats = calculate_comparison_stats(submitted_yields, challenge.reference_yields, parities)

    submission = Submission(
        challenge_id=challenge_id,
        submission_type="own_method",
        organization=organization,
        country=country,
        calculation_method=calculation_method,
        notes=notes,
        submitted_yields=submitted_yields,
        bovi_yields=bovi_yields,
        stats=stats,
        failed_cow_ids=failed_ids,
    )
    session.add(submission)
    await session.commit()
    await session.refresh(submission)
    return submission


@router.get("/submissions", response_model=list[SubmissionRead])
async def list_submissions(
    session: AsyncSession = Depends(get_session),
) -> list[Submission]:
    """List all submissions, newest first."""
    result = await session.execute(
        select(Submission).order_by(Submission.created_at.desc()).limit(100)
    )
    return list(result.scalars().all())


@router.get("/submissions/{submission_id}", response_model=SubmissionRead)
async def get_submission(
    submission_id: int,
    session: AsyncSession = Depends(get_session),
) -> Submission:
    """Get a single submission with stats."""
    sub = await session.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    return sub
```

- [ ] **Step 3: Run tests**

```bash
cd apps/backend/api && uv run pytest tests/test_benchmark_routes.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add apps/backend/api/src/bovi_api/routes/benchmark.py \
        apps/backend/api/tests/test_benchmark_routes.py
git commit -m "feat(benchmark): add submission endpoints (Pad A bovi model + Pad B CSV upload)"
```

---

## Task 8: PDF report generation and report endpoint

**Files:**
- Create: `apps/backend/api/src/bovi_api/benchmark_pdf.py`
- Modify: `apps/backend/api/src/bovi_api/routes/benchmark.py`

- [ ] **Step 1: Write a failing test**

```python
# Add to test_benchmark_routes.py or create test_benchmark_pdf.py:

def test_generate_report_pdf_returns_bytes():
    from bovi_api.benchmark_pdf import generate_report_pdf

    stats = {
        "overall": {"pearson": 0.97, "rmse": 45.0, "mae": 38.0, "mape": 4.3, "n": 100},
        "by_parity": {
            "1": {"pearson": 0.96, "rmse": 48.0, "mae": 40.0, "mape": 4.8, "n": 50},
            "3+": {"pearson": 0.98, "rmse": 42.0, "mae": 36.0, "mape": 3.9, "n": 50},
        },
        "failed_count": 0,
    }
    submitted = {str(i): 8000.0 + i * 10 for i in range(20)}
    reference = {str(i): 8000.0 + i * 12 for i in range(20)}
    bovi_y = {str(i): 8000.0 + i * 11 for i in range(20)}

    result = generate_report_pdf(
        stats=stats,
        submitted_yields=submitted,
        reference_yields=reference,
        bovi_yields=bovi_y,
        flavor="all",
        challenge_dataset="aurora",
        challenge_size="small",
    )
    assert isinstance(result, bytes)
    assert len(result) > 1000  # non-trivial PDF
    assert result[:4] == b"%PDF"
```

- [ ] **Step 2: Run failing test**

```bash
cd apps/backend/api && uv run pytest tests/ -k "generate_report_pdf" -v
```

Expected: ImportError.

- [ ] **Step 3: Implement generate_report_pdf()**

```python
# apps/backend/api/src/bovi_api/benchmark_pdf.py
"""PDF report generation for benchmark submissions."""

from __future__ import annotations

import io
from typing import Literal

from fpdf import FPDF

Flavor = Literal["icar", "bovi", "all"]


def _stats_table(pdf: FPDF, stats: dict, title: str) -> None:
    """Render a stats table for overall and per-parity breakdowns."""
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, title, ln=True)
    pdf.set_font("Helvetica", "", 9)
    headers = ["Group", "N", "Pearson", "RMSE (kg)", "MAE (kg)", "MAPE (%)"]
    col_widths = [30, 15, 25, 25, 25, 25]

    pdf.set_fill_color(220, 220, 220)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()

    def _row(label: str, s: dict) -> None:
        if s.get("n", 0) == 0:
            return
        row = [
            label,
            str(s["n"]),
            f"{s['pearson']:.4f}" if s["pearson"] is not None else "—",
            f"{s['rmse']:.1f}" if s["rmse"] is not None else "—",
            f"{s['mae']:.1f}" if s["mae"] is not None else "—",
            f"{s['mape']:.2f}" if s["mape"] is not None else "—",
        ]
        for val, w in zip(row, col_widths):
            pdf.cell(w, 6, val, border=1)
        pdf.ln()

    _row("Overall", stats["overall"])
    for parity_key in sorted(stats.get("by_parity", {}).keys()):
        _row(f"Parity {parity_key}", stats["by_parity"][parity_key])
    pdf.ln(4)


def generate_report_pdf(
    stats: dict,
    submitted_yields: dict[str, float],
    reference_yields: dict[str, float],
    bovi_yields: dict[str, float],
    flavor: Flavor = "all",
    challenge_dataset: str = "",
    challenge_size: str = "",
) -> bytes:
    """Generate a PDF report comparing submitted yields against reference/bovi values.

    Args:
        stats: Output of calculate_comparison_stats (submitted vs reference).
        submitted_yields: {cow_id: float} — user or Pad A yields.
        reference_yields: {cow_id: float} — ICAR TIM reference.
        bovi_yields: {cow_id: float} — bovi TIM yields.
        flavor: Which comparisons to include ('icar', 'bovi', or 'all').
        challenge_dataset: e.g. 'aurora' (for title).
        challenge_size: e.g. 'small' (for title).

    Returns:
        PDF bytes.

    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Bovi Benchmark Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Dataset: {challenge_dataset} / {challenge_size}", ln=True)
    pdf.cell(0, 6, f"Cows evaluated: {stats['overall']['n']}", ln=True)
    if stats.get("failed_count"):
        pdf.cell(0, 6, f"Excluded (parse failures): {stats['failed_count']}", ln=True)
    pdf.ln(6)

    if flavor in ("icar", "all"):
        _stats_table(pdf, stats, "Submitted vs ICAR Reference (TIM)")

    if flavor in ("bovi", "all"):
        # Recompute stats for submitted vs bovi_yields
        from bovi_api.benchmark_stats import calculate_comparison_stats
        # Use a flat parities dict since we don't need parity split here
        flat_parities = {cid: 1 for cid in submitted_yields}
        bovi_stats = calculate_comparison_stats(submitted_yields, bovi_yields, flat_parities)
        _stats_table(pdf, bovi_stats, "Submitted vs Bovi TIM")

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "Generated by Bovi — dairy analytics platform", align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
```

- [ ] **Step 4: Add report endpoint to benchmark.py**

Append to `benchmark.py`:

```python
from bovi_api.benchmark_pdf import Flavor, generate_report_pdf


@router.get("/submissions/{submission_id}/report")
async def download_report(
    submission_id: int,
    flavor: Flavor = "all",
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Download a PDF report for a submission."""
    sub = await session.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    challenge = await session.get(Challenge, sub.challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")

    pdf_bytes = generate_report_pdf(
        stats=sub.stats,
        submitted_yields=sub.submitted_yields,
        reference_yields=challenge.reference_yields,
        bovi_yields=sub.bovi_yields,
        flavor=flavor,
        challenge_dataset=challenge.dataset,
        challenge_size=challenge.size,
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=benchmark_report_{submission_id}.pdf"
        },
    )
```

- [ ] **Step 5: Run all backend tests**

```bash
cd apps/backend/api && uv run pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/backend/api/src/bovi_api/benchmark_pdf.py \
        apps/backend/api/src/bovi_api/routes/benchmark.py
git commit -m "feat(benchmark): add PDF report generation and download endpoint"
```

---

## Task 9: Frontend API contract (Zod schemas + api-client functions)

**Files:**
- Modify: `apps/frontend/dashboard/src/types/api.ts`
- Modify: `apps/frontend/dashboard/src/lib/api-client.ts`

- [ ] **Step 1: Add Zod schemas to types/api.ts**

Append to the end of `types/api.ts`:

```typescript
/* ------------------------------------------------------------------ */
/*  Benchmark — Challenges                                             */
/* ------------------------------------------------------------------ */

export const ChallengeReadSchema = z.object({
  id: z.number(),
  dataset: z.string(),
  size: z.string(),
  period: z.string(),
  user_id: z.string().nullable(),
  created_at: z.string().nullable(),
});
export type ChallengeRead = z.infer<typeof ChallengeReadSchema>;

export const ChallengeListSchema = z.array(ChallengeReadSchema);

export const ChallengeCreateSchema = z.object({
  dataset: z.enum(["aurora", "sunnyside"]),
  size: z.enum(["small", "medium"]),
  period: z.enum(["recent", "old", "mixed"]),
});
export type ChallengeCreate = z.infer<typeof ChallengeCreateSchema>;

/* ------------------------------------------------------------------ */
/*  Benchmark — Submissions                                            */
/* ------------------------------------------------------------------ */

export const ParityStatsSchema = z.object({
  pearson: z.number().nullable(),
  rmse: z.number().nullable(),
  mae: z.number().nullable(),
  mape: z.number().nullable(),
  n: z.number(),
});

export const ComparisonStatsSchema = z.object({
  overall: ParityStatsSchema,
  by_parity: z.record(z.string(), ParityStatsSchema),
  failed_count: z.number(),
});
export type ComparisonStats = z.infer<typeof ComparisonStatsSchema>;

export const SubmissionReadSchema = z.object({
  id: z.number(),
  challenge_id: z.number(),
  submission_type: z.string(),
  model_type: z.string().nullable(),
  organization: z.string().nullable(),
  country: z.string().nullable(),
  calculation_method: z.string().nullable(),
  notes: z.string().nullable(),
  user_id: z.string().nullable(),
  stats: ComparisonStatsSchema,
  failed_cow_ids: z.array(z.string()),
  created_at: z.string().nullable(),
});
export type SubmissionRead = z.infer<typeof SubmissionReadSchema>;

export const SubmissionListSchema = z.array(SubmissionReadSchema);
```

- [ ] **Step 2: Add API client functions to api-client.ts**

Append to `api-client.ts`:

```typescript
/* ------------------------------------------------------------------ */
/*  Benchmark — Challenges                                             */
/* ------------------------------------------------------------------ */
import type { ChallengeCreate, ChallengeRead, SubmissionRead } from "@/types/api";
import {
  ChallengeListSchema,
  ChallengeReadSchema,
  SubmissionListSchema,
  SubmissionReadSchema,
} from "@/types/api";

export async function createChallenge(data: ChallengeCreate): Promise<ChallengeRead> {
  return apiFetch("/benchmark/challenges", ChallengeReadSchema, data);
}

export async function listChallenges(): Promise<ChallengeRead[]> {
  return apiGet("/benchmark/challenges", ChallengeListSchema);
}

export async function getChallenge(id: number): Promise<ChallengeRead> {
  return apiGet(`/benchmark/challenges/${id}`, ChallengeReadSchema);
}

export function exportChallengeUrl(id: number): string {
  return `${getApiBaseUrl()}/benchmark/challenges/${id}/export`;
}

/* ------------------------------------------------------------------ */
/*  Benchmark — Submissions                                            */
/* ------------------------------------------------------------------ */

export async function submitBoviModel(
  challengeId: number,
  data: { submission_type: "bovi_model"; model_type: string; organization?: string; country?: string; notes?: string }
): Promise<SubmissionRead> {
  return apiFetch(`/benchmark/challenges/${challengeId}/submissions`, SubmissionReadSchema, data);
}

export async function submitOwnMethod(
  challengeId: number,
  file: File,
  meta: { organization?: string; country?: string; calculation_method?: string; notes?: string }
): Promise<SubmissionRead> {
  const formData = new FormData();
  formData.append("file", file);
  Object.entries(meta).forEach(([k, v]) => { if (v) formData.append(k, v); });
  const response = await fetch(`${getApiBaseUrl()}/benchmark/challenges/${challengeId}/submissions/upload`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`Upload error ${response.status}: ${JSON.stringify(error)}`);
  }
  return SubmissionReadSchema.parse(await response.json());
}

export async function listSubmissions(): Promise<SubmissionRead[]> {
  return apiGet("/benchmark/submissions", SubmissionListSchema);
}

export async function getSubmission(id: number): Promise<SubmissionRead> {
  return apiGet(`/benchmark/submissions/${id}`, SubmissionReadSchema);
}

export function downloadReportUrl(id: number, flavor: "icar" | "bovi" | "all" = "all"): string {
  return `${getApiBaseUrl()}/benchmark/submissions/${id}/report?flavor=${flavor}`;
}
```

- [ ] **Step 3: Fix TypeScript type errors**

```bash
cd apps/frontend/dashboard && bun run tsc --noEmit
```

Fix any type errors. Common issue: `exportChallengeUrl` and `downloadReportUrl` return `string` — remove the `: string` return type annotation from function declarations (TypeScript infers it).

- [ ] **Step 4: Commit**

```bash
git add apps/frontend/dashboard/src/types/api.ts \
        apps/frontend/dashboard/src/lib/api-client.ts
git commit -m "feat(benchmark): add Zod schemas and API client functions for benchmark tab"
```

---

## Task 10: Navigation + React Query hooks

**Files:**
- Modify: `apps/frontend/dashboard/src/components/dashboard/navigation.ts`
- Create: `apps/frontend/dashboard/src/app/(dashboard)/benchmark/hooks/use-challenges.ts`
- Create: `apps/frontend/dashboard/src/app/(dashboard)/benchmark/hooks/use-submissions.ts`

- [ ] **Step 1: Add /benchmark to navigation**

In `navigation.ts`:

```typescript
import { BarChart3, FlaskConical, Home, Trophy } from "lucide-react";

export const DASHBOARD_NAVIGATION: readonly NavigationItem[] = [
  { label: "Home", href: "/", icon: Home },
  { label: "Herd Stats", href: "/herd-stats", icon: BarChart3 },
  { label: "Curves", href: "/curves", icon: FlaskConical },
  { label: "Benchmark", href: "/benchmark", icon: Trophy },
];
```

- [ ] **Step 2: Create use-challenges.ts**

```typescript
// apps/frontend/dashboard/src/app/(dashboard)/benchmark/hooks/use-challenges.ts
"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createChallenge, listChallenges } from "@/lib/api-client";
import type { ChallengeCreate } from "@/types/api";

const KEY = ["benchmark-challenges"] as const;

export function useChallenges() {
  return useQuery({ queryKey: KEY, queryFn: listChallenges });
}

export function useCreateChallenge() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: ChallengeCreate) => createChallenge(data),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}
```

- [ ] **Step 3: Create use-submissions.ts**

```typescript
// apps/frontend/dashboard/src/app/(dashboard)/benchmark/hooks/use-submissions.ts
"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getSubmission, listSubmissions, submitBoviModel, submitOwnMethod } from "@/lib/api-client";

const KEY = ["benchmark-submissions"] as const;

export function useSubmissions() {
  return useQuery({ queryKey: KEY, queryFn: listSubmissions });
}

export function useSubmission(id: number) {
  return useQuery({
    queryKey: [...KEY, id],
    queryFn: () => getSubmission(id),
    enabled: !!id,
  });
}

export function useSubmitBoviModel(challengeId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Parameters<typeof submitBoviModel>[1]) =>
      submitBoviModel(challengeId, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}

export function useSubmitOwnMethod(challengeId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ file, meta }: { file: File; meta: Parameters<typeof submitOwnMethod>[2] }) =>
      submitOwnMethod(challengeId, file, meta),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}
```

- [ ] **Step 4: Type-check**

```bash
cd apps/frontend/dashboard && bun run tsc --noEmit
```

- [ ] **Step 5: Commit**

```bash
git add apps/frontend/dashboard/src/components/dashboard/navigation.ts \
        apps/frontend/dashboard/src/app/(dashboard)/benchmark/hooks/
git commit -m "feat(benchmark): add navigation entry and React Query hooks"
```

---

## Task 11: Challenge list page and card component

**Files:**
- Create: `apps/frontend/dashboard/src/app/(dashboard)/benchmark/page.tsx`
- Create: `apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/challenge-card.tsx`

- [ ] **Step 1: Create challenge-card.tsx**

```typescript
// apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/challenge-card.tsx
"use client";

import type { ReactElement } from "react";
import { Badge, Button, Card, Group, Stack, Text } from "@mantine/core";
import { useRouter } from "next/navigation";
import type { ChallengeRead } from "@/types/api";

interface Props {
  challenge: ChallengeRead;
}

export function ChallengeCard({ challenge }: Props): ReactElement {
  const router = useRouter();
  return (
    <Card shadow="sm" padding="md" radius="md" withBorder>
      <Stack gap="xs">
        <Group justify="space-between">
          <Text fw={600} size="sm">Challenge #{challenge.id}</Text>
          <Badge size="xs" variant="light">{challenge.dataset}</Badge>
        </Group>
        <Text size="xs" c="dimmed">
          {challenge.size} · {challenge.period}
          {challenge.created_at
            ? ` · ${new Date(challenge.created_at).toLocaleDateString()}`
            : ""}
        </Text>
        <Button
          size="xs"
          variant="light"
          onClick={() => router.push(`/benchmark/${challenge.id}`)}
        >
          View &amp; Submit
        </Button>
      </Stack>
    </Card>
  );
}
```

- [ ] **Step 2: Create benchmark/page.tsx**

```typescript
// apps/frontend/dashboard/src/app/(dashboard)/benchmark/page.tsx
"use client";

import type { ReactElement } from "react";
import { Button, Grid, Group, Loader, Stack, Text } from "@mantine/core";
import { Plus } from "lucide-react";
import { useRouter } from "next/navigation";
import { ChallengeCard } from "./components/challenge-card";
import { useChallenges } from "./hooks/use-challenges";

export default function BenchmarkPage(): ReactElement {
  const router = useRouter();
  const { data: challenges, isLoading, error } = useChallenges();

  if (isLoading) return <Loader />;
  if (error) return <Text c="red">Failed to load challenges.</Text>;

  return (
    <div className="space-y-6 p-6">
      <Group justify="space-between" align="center">
        <Stack gap={2}>
          <h1 className="text-2xl font-semibold">Benchmark</h1>
          <Text size="sm" c="dimmed">
            Create a challenge from a preset dataset, submit your 305-day yield calculations, and
            compare against ICAR reference values.
          </Text>
        </Stack>
        <Button leftSection={<Plus size={14} />} onClick={() => router.push("/benchmark/new")}>
          New Challenge
        </Button>
      </Group>

      {challenges && challenges.length === 0 && (
        <Text c="dimmed" size="sm">
          No challenges yet. Create one to get started.
        </Text>
      )}

      <Grid>
        {challenges?.map((c) => (
          <Grid.Col key={c.id} span={{ base: 12, sm: 6, md: 4 }}>
            <ChallengeCard challenge={c} />
          </Grid.Col>
        ))}
      </Grid>
    </div>
  );
}
```

- [ ] **Step 3: Start the dashboard and verify the Benchmark tab appears**

```bash
just run-dashboard
```

Open http://localhost:3000/benchmark — should show "No challenges yet" with a "New Challenge" button. Check browser console for errors.

- [ ] **Step 4: Commit**

```bash
git add apps/frontend/dashboard/src/app/(dashboard)/benchmark/page.tsx \
        apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/challenge-card.tsx
git commit -m "feat(benchmark): add challenge list page and card component"
```

---

## Task 12: New challenge page

**Files:**
- Create: `apps/frontend/dashboard/src/app/(dashboard)/benchmark/new/page.tsx`

- [ ] **Step 1: Create new/page.tsx**

```typescript
// apps/frontend/dashboard/src/app/(dashboard)/benchmark/new/page.tsx
"use client";

import type { ReactElement } from "react";
import { Button, Group, Loader, Select, SegmentedControl, Stack, Text } from "@mantine/core";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useCreateChallenge } from "../hooks/use-challenges";

export default function NewChallengePage(): ReactElement {
  const router = useRouter();
  const [dataset, setDataset] = useState<"aurora" | "sunnyside">("aurora");
  const [size, setSize] = useState<"small" | "medium">("small");
  const [period, setPeriod] = useState<"recent" | "old" | "mixed">("recent");

  const { mutate, isPending, error } = useCreateChallenge();

  function handleCreate() {
    mutate(
      { dataset, size, period },
      {
        onSuccess: (c) => router.push(`/benchmark/${c.id}`),
      }
    );
  }

  return (
    <div className="max-w-lg space-y-6 p-6">
      <Stack gap={2}>
        <h1 className="text-2xl font-semibold">New Challenge</h1>
        <Text size="sm" c="dimmed">
          Choose a dataset, size, and period. The system will sample cows and compute reference
          305-day yields via TIM.
        </Text>
      </Stack>

      <Stack gap="md">
        <Select
          label="Dataset"
          data={[
            { value: "aurora", label: "Aurora Ridge" },
            { value: "sunnyside", label: "Sunnyside" },
          ]}
          value={dataset}
          onChange={(v) => v && setDataset(v as typeof dataset)}
        />

        <div>
          <Text size="sm" fw={500} mb={4}>Size</Text>
          <SegmentedControl
            value={size}
            onChange={(v) => setSize(v as typeof size)}
            data={[
              { value: "small", label: "Small (200 cows)" },
              { value: "medium", label: "Medium (1000 cows)" },
            ]}
          />
        </div>

        <div>
          <Text size="sm" fw={500} mb={4}>Period</Text>
          <SegmentedControl
            value={period}
            onChange={(v) => setPeriod(v as typeof period)}
            data={[
              { value: "recent", label: "Recent" },
              { value: "old", label: "Old" },
              { value: "mixed", label: "Mixed" },
            ]}
          />
        </div>

        {error && <Text c="red" size="sm">{(error as Error).message}</Text>}

        <Group>
          <Button onClick={handleCreate} loading={isPending}>
            Create Challenge
          </Button>
          <Button variant="subtle" onClick={() => router.back()}>
            Cancel
          </Button>
        </Group>

        {isPending && (
          <Group gap="xs">
            <Loader size="xs" />
            <Text size="xs" c="dimmed">Computing reference yields — this may take 30–60 seconds…</Text>
          </Group>
        )}
      </Stack>
    </div>
  );
}
```

- [ ] **Step 2: Test in browser**

Navigate to http://localhost:3000/benchmark/new — form renders, clicking "Create Challenge" calls the API.

- [ ] **Step 3: Commit**

```bash
git add apps/frontend/dashboard/src/app/(dashboard)/benchmark/new/page.tsx
git commit -m "feat(benchmark): add new challenge creation page"
```

---

## Task 13: Challenge detail page with submission forms and results

**Files:**
- Create: `apps/frontend/dashboard/src/app/(dashboard)/benchmark/[id]/page.tsx`
- Create: `apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/submission-form-bovi.tsx`
- Create: `apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/submission-form-upload.tsx`
- Create: `apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/submission-form.tsx`
- Create: `apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/comparison-results.tsx`

- [ ] **Step 1: Create submission-form-bovi.tsx (Pad A)**

```typescript
// apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/submission-form-bovi.tsx
"use client";

import type { ReactElement } from "react";
import { Button, Select, Stack, Text, TextInput } from "@mantine/core";
import { useState } from "react";
import { useSubmitBoviModel } from "../hooks/use-submissions";

interface Props {
  challengeId: number;
  onSuccess: () => void;
}

export function SubmissionFormBovi({ challengeId, onSuccess }: Props): ReactElement {
  const [modelType, setModelType] = useState("tim");
  const [organization, setOrganization] = useState("");
  const [country, setCountry] = useState("");
  const { mutate, isPending, error } = useSubmitBoviModel(challengeId);

  function handleSubmit() {
    mutate(
      { submission_type: "bovi_model", model_type: modelType, organization, country },
      { onSuccess }
    );
  }

  return (
    <Stack gap="sm">
      <Text size="sm" c="dimmed">
        Bovi will compute 305-day yields for all challenge cows using the selected model.
      </Text>
      <Select
        label="Model"
        data={[{ value: "tim", label: "TIM (ICAR Test Interval Method)" }]}
        value={modelType}
        onChange={(v) => v && setModelType(v)}
      />
      <TextInput label="Organization" value={organization} onChange={(e) => setOrganization(e.target.value)} />
      <TextInput label="Country" value={country} onChange={(e) => setCountry(e.target.value)} />
      {error && <Text c="red" size="xs">{(error as Error).message}</Text>}
      <Button onClick={handleSubmit} loading={isPending}>Run &amp; Submit</Button>
    </Stack>
  );
}
```

- [ ] **Step 2: Create submission-form-upload.tsx (Pad B)**

```typescript
// apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/submission-form-upload.tsx
"use client";

import type { ReactElement } from "react";
import { Button, FileInput, Stack, Text, TextInput } from "@mantine/core";
import { useState } from "react";
import { exportChallengeUrl } from "@/lib/api-client";
import { useSubmitOwnMethod } from "../hooks/use-submissions";

interface Props {
  challengeId: number;
  onSuccess: () => void;
}

export function SubmissionFormUpload({ challengeId, onSuccess }: Props): ReactElement {
  const [file, setFile] = useState<File | null>(null);
  const [organization, setOrganization] = useState("");
  const [country, setCountry] = useState("");
  const [calcMethod, setCalcMethod] = useState("");
  const { mutate, isPending, error } = useSubmitOwnMethod(challengeId);

  function handleSubmit() {
    if (!file) return;
    mutate({ file, meta: { organization, country, calculation_method: calcMethod } }, { onSuccess });
  }

  return (
    <Stack gap="sm">
      <Text size="sm" c="dimmed">
        Download the test data, calculate 305-day yields with your own method, then upload your
        results as a CSV with columns: <code>cow_id, yield_305day</code>.
      </Text>
      <Button
        variant="outline"
        component="a"
        href={exportChallengeUrl(challengeId)}
        download
      >
        Download test data CSV
      </Button>
      <FileInput
        label="Upload results CSV"
        accept=".csv"
        value={file}
        onChange={setFile}
        placeholder="challenge_results.csv"
      />
      <TextInput label="Organization" value={organization} onChange={(e) => setOrganization(e.target.value)} />
      <TextInput label="Country" value={country} onChange={(e) => setCountry(e.target.value)} />
      <TextInput label="Calculation method" value={calcMethod} onChange={(e) => setCalcMethod(e.target.value)} />
      {error && <Text c="red" size="xs">{(error as Error).message}</Text>}
      <Button onClick={handleSubmit} loading={isPending} disabled={!file}>Submit</Button>
    </Stack>
  );
}
```

- [ ] **Step 3: Create submission-form.tsx (Mantine Tabs wrapper)**

```typescript
// apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/submission-form.tsx
"use client";

import type { ReactElement } from "react";
import { Tabs } from "@mantine/core";
import { SubmissionFormBovi } from "./submission-form-bovi";
import { SubmissionFormUpload } from "./submission-form-upload";

interface Props {
  challengeId: number;
  onSuccess: () => void;
}

export function SubmissionForm({ challengeId, onSuccess }: Props): ReactElement {
  return (
    <Tabs defaultValue="bovi">
      <Tabs.List>
        <Tabs.Tab value="bovi">Pad A — Bovi model</Tabs.Tab>
        <Tabs.Tab value="upload">Pad B — Own method</Tabs.Tab>
      </Tabs.List>
      <Tabs.Panel value="bovi" pt="sm">
        <SubmissionFormBovi challengeId={challengeId} onSuccess={onSuccess} />
      </Tabs.Panel>
      <Tabs.Panel value="upload" pt="sm">
        <SubmissionFormUpload challengeId={challengeId} onSuccess={onSuccess} />
      </Tabs.Panel>
    </Tabs>
  );
}
```

- [ ] **Step 4: Create comparison-results.tsx**

```typescript
// apps/frontend/dashboard/src/app/(dashboard)/benchmark/components/comparison-results.tsx
"use client";

import type { ReactElement } from "react";
import { Badge, Button, Group, SegmentedControl, Stack, Table, Text } from "@mantine/core";
import { useState } from "react";
import { downloadReportUrl } from "@/lib/api-client";
import type { SubmissionRead } from "@/types/api";

interface Props {
  submission: SubmissionRead;
}

function StatsRow({ label, stats }: { label: string; stats: Record<string, number | null> }): ReactElement {
  const fmt = (v: number | null) => (v === null ? "—" : v.toFixed(3));
  return (
    <Table.Tr>
      <Table.Td>{label}</Table.Td>
      <Table.Td>{stats.n}</Table.Td>
      <Table.Td>{fmt(stats.pearson)}</Table.Td>
      <Table.Td>{fmt(stats.rmse)} kg</Table.Td>
      <Table.Td>{fmt(stats.mae)} kg</Table.Td>
      <Table.Td>{fmt(stats.mape)}%</Table.Td>
    </Table.Tr>
  );
}

export function ComparisonResults({ submission }: Props): ReactElement {
  const [flavor, setFlavor] = useState<"icar" | "bovi" | "all">("all");
  const { stats } = submission;

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Text fw={600}>Results</Text>
        <Badge variant="light" color="green">
          {submission.submission_type === "bovi_model"
            ? `Bovi — ${submission.model_type}`
            : "Own method"}
        </Badge>
      </Group>

      <Table withTableBorder withColumnBorders>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Group</Table.Th>
            <Table.Th>N</Table.Th>
            <Table.Th>Pearson</Table.Th>
            <Table.Th>RMSE</Table.Th>
            <Table.Th>MAE</Table.Th>
            <Table.Th>MAPE</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          <StatsRow label="Overall" stats={stats.overall} />
          {Object.entries(stats.by_parity).map(([p, s]) => (
            <StatsRow key={p} label={`Parity ${p}`} stats={s} />
          ))}
        </Table.Tbody>
      </Table>

      {stats.failed_count > 0 && (
        <Text size="xs" c="dimmed">{stats.failed_count} cow(s) excluded from stats.</Text>
      )}

      <Group>
        <Text size="sm" fw={500}>Report flavor:</Text>
        <SegmentedControl
          value={flavor}
          onChange={(v) => setFlavor(v as typeof flavor)}
          data={[
            { value: "icar", label: "vs ICAR" },
            { value: "bovi", label: "vs Bovi" },
            { value: "all", label: "Full" },
          ]}
          size="xs"
        />
        <Button
          size="xs"
          component="a"
          href={downloadReportUrl(submission.id, flavor)}
          download
        >
          Download PDF
        </Button>
      </Group>
    </Stack>
  );
}
```

- [ ] **Step 5: Create [id]/page.tsx**

```typescript
// apps/frontend/dashboard/src/app/(dashboard)/benchmark/[id]/page.tsx
"use client";

import type { ReactElement } from "react";
import { Badge, Card, Divider, Group, Loader, Stack, Text } from "@mantine/core";
import { useParams } from "next/navigation";
import { useState } from "react";
import { ComparisonResults } from "../components/comparison-results";
import { SubmissionForm } from "../components/submission-form";
import { useSubmissions } from "../hooks/use-submissions";

export default function ChallengeDetailPage(): ReactElement {
  const { id } = useParams<{ id: string }>();
  const challengeId = Number(id);
  const { data: submissions, isLoading, refetch } = useSubmissions();
  const [submitted, setSubmitted] = useState(false);

  const challengeSubmissions = submissions?.filter((s) => s.challenge_id === challengeId) ?? [];
  const latest = challengeSubmissions[0];

  if (isLoading) return <Loader />;

  return (
    <div className="space-y-6 p-6">
      <Stack gap={2}>
        <Group>
          <h1 className="text-2xl font-semibold">Challenge #{challengeId}</h1>
        </Group>
        <Text size="sm" c="dimmed">
          Submit your 305-day yield calculations using a Bovi model (Pad A) or upload your own CSV
          (Pad B). Then download the comparison report.
        </Text>
      </Stack>

      <Card withBorder padding="md">
        <Text fw={600} mb="sm">Submit Results</Text>
        <SubmissionForm
          challengeId={challengeId}
          onSuccess={() => { setSubmitted(true); refetch(); }}
        />
      </Card>

      {latest && (
        <>
          <Divider />
          <Card withBorder padding="md">
            <ComparisonResults submission={latest} />
          </Card>
        </>
      )}

      {challengeSubmissions.length > 1 && (
        <Text size="xs" c="dimmed">{challengeSubmissions.length} submissions total for this challenge.</Text>
      )}
    </div>
  );
}
```

- [ ] **Step 6: Test end-to-end in browser**

With both API and dashboard running (`just run-api` + `just run-dashboard`):
1. Go to http://localhost:3000/benchmark
2. Click "New Challenge" → select Aurora/small/recent → "Create Challenge"
3. On challenge detail page: submit with Pad A (TIM model)
4. Verify stats table renders
5. Click "Download PDF" → PDF downloads

- [ ] **Step 7: Commit**

```bash
git add apps/frontend/dashboard/src/app/(dashboard)/benchmark/
git commit -m "feat(benchmark): add challenge detail page with submission forms and results display"
```

---

## Task 14: Final cleanup and full test run

- [ ] **Step 1: Run all backend tests**

```bash
cd apps/backend/api && uv run pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run frontend type-check**

```bash
cd apps/frontend/dashboard && bun run tsc --noEmit
```

Expected: no type errors.

- [ ] **Step 3: Run full lint**

```bash
just lint
```

Fix any ruff or basedpyright errors.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(benchmark): complete ICAR accreditation benchmark tab integration"
```
