# Herd Stats Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add named herd profile CRUD (Phase 1) and CSV upload with normalization (Phase 2) so users can save and reuse herd stats on the curves page.

**Architecture:** SQLModel `HerdProfile` table in the Central API; a new FastAPI router for CRUD + stateless CSV preview; normalization logic inlined in the ingestion utility (no cross-package ML imports); a `HerdStatsRangeNormalizationTransform` utility class (not in TransformRegistry) in the lactation-autoencoder package; a new Next.js `/herd-stats` management page; a profile-selector dropdown wired into the existing `AutoencoderInputPanel`.

**Tech Stack:** Python 3.12 · FastAPI · SQLModel · Alembic · asyncpg · aiosqlite (tests) · Next.js · Mantine · Zod · React Query · uv · bun

**Spec:** `docs/superpowers/specs/2026-04-15-herd-stats-management-design.md`

---

## File Map

### Phase 1 — CRUD + manual entry

**Create:**
- `apps/backend/api/src/bovi_api/routes/herd_profiles.py` — CRUD router
- `apps/backend/api/tests/conftest.py` — pytest fixtures (TestClient + in-memory SQLite)
- `apps/backend/api/tests/test_herd_profiles.py` — CRUD + upload tests
- `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/page.tsx`
- `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/components/herd-profile-list.tsx`
- `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/components/herd-profile-form.tsx`
- `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/hooks/use-herd-profiles.ts`
- `apps/frontend/dashboard/src/lib/herd-profile-utils.ts`

**Modify:**
- `apps/backend/api/src/bovi_api/models.py` — add `HerdProfile*` SQLModel classes
- `apps/backend/api/alembic/env.py` — import `HerdProfile` to register table
- `apps/backend/api/alembic/versions/<hash>_add_herd_profiles_table.py` — generated + trigger
- `apps/backend/api/src/bovi_api/app.py` — include herd_profiles router
- `apps/backend/api/pyproject.toml` — add `aiosqlite` test dep
- `apps/frontend/dashboard/src/types/api.ts` — Zod schemas for HerdProfile
- `apps/frontend/dashboard/src/lib/api-client.ts` — add GET/PUT/DELETE helpers + herd profile functions
- `apps/frontend/dashboard/src/components/dashboard/navigation.ts` — Herd Stats nav item
- `apps/frontend/dashboard/src/app/(dashboard)/curves/components/autoencoder-input-panel.tsx` — profile dropdown

### Phase 2 — CSV upload

**Create:**
- `apps/backend/api/src/bovi_api/herd_stats_ingestion.py` — CSV parsing + inline normalization
- `apps/backend/api/tests/test_herd_stats_ingestion.py` — ingestion unit tests
- `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/hooks/use-herd-profile-upload.ts`
- `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/components/herd-profile-upload.tsx`

**Modify:**
- `packages/models/lactation-autoencoder/src/lactation_autoencoder/dataloaders/transforms/lactation_transforms.py` — add `HerdStatsRangeNormalizationTransform`
- `packages/models/lactation-autoencoder/data/experiments/lactation_autoencoder/versions/v15/config/config.yaml` — add `herd_stats_ranges`
- `packages/models/lactation-autoencoder/tests/test_lactation_transforms.py` — add transform tests
- `apps/backend/api/src/bovi_api/routes/herd_profiles.py` — add `csv-preview` endpoint
- `apps/frontend/dashboard/src/types/api.ts` — add `HerdProfileUploadResponseSchema`
- `apps/frontend/dashboard/src/lib/api-client.ts` — add `uploadHerdProfileCsv`
- `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/page.tsx` — wire upload component

---

## Phase 1 — CRUD + Manual Entry

---

### Task 1: Add `HerdProfile` SQLModel

**Files:**
- Modify: `apps/backend/api/src/bovi_api/models.py`

- [ ] **Step 1: Add imports and model classes**

Open `apps/backend/api/src/bovi_api/models.py`. Add after the existing imports:

```python
from sqlalchemy import Column, DateTime, UniqueConstraint
from sqlalchemy import func as sa_func
```

Add after the existing `FittingResultRead` class:

```python
class HerdProfileBase(SQLModel):
    """Shared fields for herd profiles."""

    name: str = Field(max_length=100, description="User-given name for this profile")
    description: str = Field(default="", max_length=500)
    achieved_21_milk: float = Field(ge=0.0, le=1.0)
    achieved_305_milk: float = Field(ge=0.0, le=1.0)
    achieved_75_milk: float = Field(ge=0.0, le=1.0)
    achieved_milk: float = Field(ge=0.0, le=1.0)
    days_dry: float = Field(ge=0.0, le=1.0)
    days_in_milk: float = Field(ge=0.0, le=1.0)
    days_open: float = Field(ge=0.0, le=1.0)
    days_pregnant: float = Field(ge=0.0, le=1.0)
    historic_calving_interval: float = Field(ge=0.0, le=1.0)
    quality_sequence: float = Field(ge=0.0, le=1.0)


class HerdProfile(HerdProfileBase, table=True):
    """Database table for user-managed herd stat profiles."""

    __tablename__ = "herd_profiles"
    __table_args__ = (UniqueConstraint("name", name="uq_herd_profile_name"),)

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=sa_func.now()),
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=sa_func.now()),
    )


class HerdProfileCreate(HerdProfileBase):
    """Request body for creating or updating a herd profile."""


class HerdProfileRead(HerdProfileBase):
    """Response body (includes auto-assigned fields; timestamps may be None in SQLite)."""

    id: int
    created_at: datetime | None  # None only when DB does not fill server default (e.g. SQLite)
    updated_at: datetime | None
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
cd apps/backend/api && uv run python -c "from bovi_api.models import HerdProfile, HerdProfileCreate, HerdProfileRead; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add apps/backend/api/src/bovi_api/models.py
git commit -m "feat: add HerdProfile SQLModel classes"
```

---

### Task 2: Tests first — write failing CRUD tests

**Files:**
- Modify: `apps/backend/api/pyproject.toml`
- Create: `apps/backend/api/tests/conftest.py`
- Create: `apps/backend/api/tests/test_herd_profiles.py`

- [ ] **Step 1: Add test dependency**

Open `apps/backend/api/pyproject.toml`. Add to `dependencies`:

```toml
    "aiosqlite>=0.20.0",
```

Run:
```bash
cd apps/backend/api && uv sync
```

- [ ] **Step 2: Create `tests/conftest.py`**

Create `apps/backend/api/tests/conftest.py`:

```python
"""Shared pytest fixtures for the bovi-api test suite."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from bovi_api.app import create_app
from bovi_api.database import get_session
from bovi_api.models import FittingResult, HerdProfile  # noqa: F401 — register tables


@pytest.fixture(scope="function")
def client():
    """TestClient backed by an in-memory SQLite database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _create_tables() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    asyncio.run(_create_tables())  # Python 3.12: use asyncio.run(), not get_event_loop()

    async def override_get_session():
        async with session_factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_session] = override_get_session

    with TestClient(app) as c:
        yield c
```

- [ ] **Step 3: Write failing tests in `tests/test_herd_profiles.py`**

Create `apps/backend/api/tests/test_herd_profiles.py`:

```python
"""Tests for the herd profiles CRUD API."""

VALID_PROFILE = {
    "name": "Test Herd",
    "description": "A test herd profile",
    "achieved_21_milk": 0.53,
    "achieved_305_milk": 0.50,
    "achieved_75_milk": 0.55,
    "achieved_milk": 0.41,
    "days_dry": 0.39,
    "days_in_milk": 0.44,
    "days_open": 0.38,
    "days_pregnant": 0.62,
    "historic_calving_interval": 0.54,
    "quality_sequence": 0.33,
}


def test_create_profile(client):
    response = client.post("/herd-profiles/", json=VALID_PROFILE)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Herd"
    assert data["id"] is not None
    assert "created_at" in data  # may be None under SQLite


def test_list_profiles_empty(client):
    response = client.get("/herd-profiles/")
    assert response.status_code == 200
    assert response.json() == []


def test_list_profiles_after_create(client):
    client.post("/herd-profiles/", json=VALID_PROFILE)
    response = client.get("/herd-profiles/")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_profile(client):
    created = client.post("/herd-profiles/", json=VALID_PROFILE).json()
    response = client.get(f"/herd-profiles/{created['id']}")
    assert response.status_code == 200
    assert response.json()["name"] == "Test Herd"


def test_get_profile_not_found(client):
    response = client.get("/herd-profiles/999")
    assert response.status_code == 404


def test_update_profile(client):
    created = client.post("/herd-profiles/", json=VALID_PROFILE).json()
    updated = {**VALID_PROFILE, "name": "Updated Herd", "achieved_21_milk": 0.70}
    response = client.put(f"/herd-profiles/{created['id']}", json=updated)
    assert response.status_code == 200
    assert response.json()["name"] == "Updated Herd"
    assert response.json()["achieved_21_milk"] == 0.70


def test_delete_profile(client):
    created = client.post("/herd-profiles/", json=VALID_PROFILE).json()
    response = client.delete(f"/herd-profiles/{created['id']}")
    assert response.status_code == 204
    assert client.get(f"/herd-profiles/{created['id']}").status_code == 404


def test_duplicate_name_returns_409(client):
    client.post("/herd-profiles/", json=VALID_PROFILE)
    response = client.post("/herd-profiles/", json=VALID_PROFILE)
    assert response.status_code == 409


def test_invalid_stat_value_returns_422(client):
    invalid = {**VALID_PROFILE, "achieved_21_milk": 1.5}  # exceeds 1.0
    response = client.post("/herd-profiles/", json=invalid)
    assert response.status_code == 422
```

- [ ] **Step 4: Run tests to verify they fail (router not yet created)**

```bash
cd apps/backend/api && uv run pytest tests/test_herd_profiles.py -v
```

Expected: tests fail — `404 Not Found` or `ImportError` because `/herd-profiles/` routes don't exist yet.

---

### Task 3: CRUD API router

**Files:**
- Create: `apps/backend/api/src/bovi_api/routes/herd_profiles.py`
- Modify: `apps/backend/api/src/bovi_api/app.py`

- [ ] **Step 1: Create the router**

Create `apps/backend/api/src/bovi_api/routes/herd_profiles.py`:

```python
"""CRUD endpoints for user-managed herd stat profiles."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from bovi_api.database import get_session
from bovi_api.models import HerdProfile, HerdProfileCreate, HerdProfileRead

router = APIRouter(tags=["herd-profiles"])


@router.get("/", response_model=list[HerdProfileRead])
async def list_herd_profiles(
    session: AsyncSession = Depends(get_session),
) -> list[HerdProfile]:
    """List all herd profiles, newest first."""
    result = await session.execute(
        select(HerdProfile).order_by(HerdProfile.created_at.desc())
    )
    return list(result.scalars().all())


@router.post("/", response_model=HerdProfileRead, status_code=201)
async def create_herd_profile(
    profile: HerdProfileCreate,
    session: AsyncSession = Depends(get_session),
) -> HerdProfile:
    """Create a new herd profile."""
    db_profile = HerdProfile(**profile.model_dump())
    session.add(db_profile)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=409, detail=f"A profile named '{profile.name}' already exists."
        )
    await session.refresh(db_profile)
    return db_profile


@router.get("/{profile_id}", response_model=HerdProfileRead)
async def get_herd_profile(
    profile_id: int,
    session: AsyncSession = Depends(get_session),
) -> HerdProfile:
    """Retrieve a single herd profile by ID."""
    profile = await session.get(HerdProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Herd profile not found")
    return profile


@router.put("/{profile_id}", response_model=HerdProfileRead)
async def update_herd_profile(
    profile_id: int,
    update: HerdProfileCreate,
    session: AsyncSession = Depends(get_session),
) -> HerdProfile:
    """Update an existing herd profile."""
    profile = await session.get(HerdProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Herd profile not found")
    for field, value in update.model_dump().items():
        setattr(profile, field, value)
    session.add(profile)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=409, detail=f"A profile named '{update.name}' already exists."
        )
    await session.refresh(profile)
    return profile


@router.delete("/{profile_id}", status_code=204)
async def delete_herd_profile(
    profile_id: int,
    session: AsyncSession = Depends(get_session),
) -> None:
    """Delete a herd profile."""
    profile = await session.get(HerdProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Herd profile not found")
    await session.delete(profile)
    await session.commit()
```

- [ ] **Step 2: Register router in `app.py`**

Open `apps/backend/api/src/bovi_api/app.py`. Change the import:

```python
from bovi_api.routes import health, herd_profiles, proxy, results
```

Add after `app.include_router(results.router)`:

```python
    app.include_router(herd_profiles.router, prefix="/herd-profiles")
```

- [ ] **Step 3: Run tests to verify they all pass**

```bash
cd apps/backend/api && uv run pytest tests/test_herd_profiles.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add apps/backend/api/src/bovi_api/routes/herd_profiles.py \
        apps/backend/api/src/bovi_api/app.py \
        apps/backend/api/tests/ \
        apps/backend/api/pyproject.toml
git commit -m "feat: add herd profiles CRUD router with tests"
```

---

### Task 4: Alembic migration

**Files:**
- Modify: `apps/backend/api/alembic/env.py`
- Create: `apps/backend/api/alembic/versions/<hash>_add_herd_profiles_table.py`

- [ ] **Step 1: Register `HerdProfile` in alembic env**

Open `apps/backend/api/alembic/env.py`. Change line 7 to:

```python
from bovi_api.models import FittingResult, HerdProfile  # noqa: F401 — registers tables
```

- [ ] **Step 2: Generate the migration**

```bash
cd apps/backend/api && uv run alembic revision --autogenerate -m "add herd_profiles table"
```

Expected: creates a file in `alembic/versions/` like `<hash>_add_herd_profiles_table.py`.

- [ ] **Step 3: Add the `updated_at` PostgreSQL trigger to the migration**

Open the newly generated migration file. In `upgrade()`, after `op.create_table(...)`, add:

```python
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    op.execute("""
        CREATE TRIGGER update_herd_profiles_updated_at
            BEFORE UPDATE ON herd_profiles
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
```

In `downgrade()`, before `op.drop_table(...)`, add:

```python
    op.execute("DROP TRIGGER IF EXISTS update_herd_profiles_updated_at ON herd_profiles;")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")
```

- [ ] **Step 4: Commit**

```bash
git add apps/backend/api/alembic/env.py apps/backend/api/alembic/versions/
git commit -m "feat: add Alembic migration for herd_profiles table with updated_at trigger"
```

---

### Task 5: Frontend — Zod types and API client

**Files:**
- Modify: `apps/frontend/dashboard/src/types/api.ts`
- Modify: `apps/frontend/dashboard/src/lib/api-client.ts`

- [ ] **Step 1: Add Zod schemas to `types/api.ts`**

At the end of `apps/frontend/dashboard/src/types/api.ts`, add:

```typescript
/* ------------------------------------------------------------------ */
/*  Herd Profiles                                                      */
/* ------------------------------------------------------------------ */

export const HerdProfileSchema = z.object({
  id: z.number(),
  name: z.string().max(100),
  description: z.string().max(500),
  achieved_21_milk: z.number().min(0).max(1),
  achieved_305_milk: z.number().min(0).max(1),
  achieved_75_milk: z.number().min(0).max(1),
  achieved_milk: z.number().min(0).max(1),
  days_dry: z.number().min(0).max(1),
  days_in_milk: z.number().min(0).max(1),
  days_open: z.number().min(0).max(1),
  days_pregnant: z.number().min(0).max(1),
  historic_calving_interval: z.number().min(0).max(1),
  quality_sequence: z.number().min(0).max(1),
  created_at: z.string().nullable(),
  updated_at: z.string().nullable(),
});

export const HerdProfileCreateSchema = HerdProfileSchema.omit({
  id: true,
  created_at: true,
  updated_at: true,
});

export const HerdProfileListSchema = z.array(HerdProfileSchema);

export type HerdProfile = z.infer<typeof HerdProfileSchema>;
export type HerdProfileCreate = z.infer<typeof HerdProfileCreateSchema>;
```

- [ ] **Step 2: Add HTTP helpers and herd profile functions to `api-client.ts`**

Add these three helpers after the existing `apiFetch` function:

```typescript
async function apiGet<T>(path: string, schema: z.ZodType<T>): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`API error ${response.status} on ${path}: ${JSON.stringify(error)}`);
  }
  const data: unknown = await response.json();
  return schema.parse(data);
}

async function apiPut<T>(path: string, schema: z.ZodType<T>, body: unknown): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`API error ${response.status} on ${path}: ${JSON.stringify(error)}`);
  }
  const data: unknown = await response.json();
  return schema.parse(data);
}

async function apiDelete(path: string): Promise<void> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, { method: "DELETE" });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`API error ${response.status} on ${path}: ${JSON.stringify(error)}`);
  }
}
```

Add imports at the top of the file (after the existing imports):

```typescript
import type { HerdProfile, HerdProfileCreate } from "@/types/api";
import { HerdProfileSchema, HerdProfileListSchema } from "@/types/api";
```

Add herd profile API functions at the end of the file:

```typescript
/* ------------------------------------------------------------------ */
/*  Herd Profiles                                                      */
/* ------------------------------------------------------------------ */

export async function listHerdProfiles(): Promise<HerdProfile[]> {
  return apiGet("/herd-profiles/", HerdProfileListSchema);
}

export async function getHerdProfile(id: number): Promise<HerdProfile> {
  return apiGet(`/herd-profiles/${id}`, HerdProfileSchema);
}

export async function createHerdProfile(data: HerdProfileCreate): Promise<HerdProfile> {
  return apiFetch("/herd-profiles/", HerdProfileSchema, data);
}

export async function updateHerdProfile(id: number, data: HerdProfileCreate): Promise<HerdProfile> {
  return apiPut(`/herd-profiles/${id}`, HerdProfileSchema, data);
}

export async function deleteHerdProfile(id: number): Promise<void> {
  return apiDelete(`/herd-profiles/${id}`);
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd apps/frontend/dashboard && bun run tsc --noEmit
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add apps/frontend/dashboard/src/types/api.ts apps/frontend/dashboard/src/lib/api-client.ts
git commit -m "feat: add HerdProfile Zod types and API client functions"
```

---

### Task 6: Frontend — field mapping utility

**Files:**
- Create: `apps/frontend/dashboard/src/lib/herd-profile-utils.ts`

- [ ] **Step 1: Create the utility**

Create `apps/frontend/dashboard/src/lib/herd-profile-utils.ts`:

```typescript
/**
 * Utilities for mapping between HerdProfile DB fields (snake_case) and the
 * number[] expected by HerdStatsForm (indexed by HERD_STATS_METADATA order).
 */

import type { HerdProfile, HerdProfileCreate } from "@/types/api";

type StatField = keyof Omit<HerdProfileCreate, "name" | "description">;

/**
 * snake_case field names in the same order as HERD_STATS_METADATA indices 0–9.
 * Index 0 = Achieved21Milk, index 9 = QualitySequence.
 */
export const HERD_PROFILE_FIELD_ORDER: StatField[] = [
  "achieved_21_milk",
  "achieved_305_milk",
  "achieved_75_milk",
  "achieved_milk",
  "days_dry",
  "days_in_milk",
  "days_open",
  "days_pregnant",
  "historic_calving_interval",
  "quality_sequence",
];

/** Convert a saved HerdProfile → number[] for HerdStatsForm. */
export function herdProfileToStats(profile: HerdProfile): number[] {
  return HERD_PROFILE_FIELD_ORDER.map((field) => profile[field] as number);
}

/** Convert number[] from HerdStatsForm → stat fields for HerdProfileCreate. */
export function statsToHerdProfileFields(stats: number[]): Record<StatField, number> {
  return Object.fromEntries(
    HERD_PROFILE_FIELD_ORDER.map((field, i) => [field, stats[i] ?? 0])
  ) as Record<StatField, number>;
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd apps/frontend/dashboard && bun run tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add apps/frontend/dashboard/src/lib/herd-profile-utils.ts
git commit -m "feat: add herd profile field mapping utilities"
```

---

### Task 7: Frontend — React Query hook

**Files:**
- Create: `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/hooks/use-herd-profiles.ts`

- [ ] **Step 1: Create the hook file** (also creates the directory)

Create `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/hooks/use-herd-profiles.ts`:

```typescript
"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createHerdProfile,
  deleteHerdProfile,
  listHerdProfiles,
  updateHerdProfile,
} from "@/lib/api-client";
import type { HerdProfileCreate } from "@/types/api";

const QUERY_KEY = ["herd-profiles"] as const;

export function useHerdProfiles() {
  return useQuery({
    queryKey: QUERY_KEY,
    queryFn: listHerdProfiles,
  });
}

export function useCreateHerdProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: HerdProfileCreate) => createHerdProfile(data),
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  });
}

export function useUpdateHerdProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: HerdProfileCreate }) =>
      updateHerdProfile(id, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  });
}

export function useDeleteHerdProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => deleteHerdProfile(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  });
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd apps/frontend/dashboard && bun run tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add apps/frontend/dashboard/src/app/(dashboard)/herd-stats/
git commit -m "feat: add useHerdProfiles React Query hooks"
```

---

### Task 8: Frontend — HerdProfileForm component

**Files:**
- Create: `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/components/herd-profile-form.tsx`

- [ ] **Step 1: Create the form component**

Create `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/components/herd-profile-form.tsx`:

```typescript
"use client";

import type { ReactElement } from "react";
import { useState } from "react";
import { Button, Group, Stack, Textarea, TextInput } from "@mantine/core";
import { HerdStatsForm } from "@/app/(dashboard)/autoencoder/components/herd-stats-form";
import { DEFAULT_HERD_STATS } from "@/data/herd-stats-metadata";
import { herdProfileToStats, statsToHerdProfileFields } from "@/lib/herd-profile-utils";
import type { HerdProfile, HerdProfileCreate } from "@/types/api";

interface HerdProfileFormProps {
  readonly initial?: HerdProfile;
  readonly onSubmit: (data: HerdProfileCreate) => void;
  readonly onCancel: () => void;
  readonly isLoading: boolean;
}

export function HerdProfileForm({
  initial,
  onSubmit,
  onCancel,
  isLoading,
}: HerdProfileFormProps): ReactElement {
  const [name, setName] = useState(initial?.name ?? "");
  const [description, setDescription] = useState(initial?.description ?? "");
  const [stats, setStats] = useState<number[]>(
    initial ? herdProfileToStats(initial) : [...DEFAULT_HERD_STATS]
  );

  function handleSubmit() {
    onSubmit({
      name,
      description,
      ...statsToHerdProfileFields(stats),
    });
  }

  return (
    <Stack gap="md">
      <TextInput
        label="Profile name"
        placeholder="e.g. High-producing Holstein"
        value={name}
        onChange={(e) => setName(e.currentTarget.value)}
        maxLength={100}
        required
      />
      <Textarea
        label="Description"
        placeholder="Optional notes about this herd"
        value={description}
        onChange={(e) => setDescription(e.currentTarget.value)}
        maxLength={500}
        autosize
        minRows={2}
      />
      <div>
        <p className="mb-3 text-xs text-muted-foreground">
          All values normalized 0–1.
        </p>
        <HerdStatsForm values={stats} onChange={setStats} />
      </div>
      <Group justify="flex-end">
        <Button variant="subtle" onClick={onCancel} disabled={isLoading}>
          Cancel
        </Button>
        <Button
          onClick={handleSubmit}
          loading={isLoading}
          disabled={!name.trim()}
          color="violet"
        >
          {initial ? "Save changes" : "Create profile"}
        </Button>
      </Group>
    </Stack>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd apps/frontend/dashboard && bun run tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add apps/frontend/dashboard/src/app/(dashboard)/herd-stats/components/herd-profile-form.tsx
git commit -m "feat: add HerdProfileForm component"
```

---

### Task 9: Frontend — HerdProfileList component

**Files:**
- Create: `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/components/herd-profile-list.tsx`

- [ ] **Step 1: Create the list component**

Create `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/components/herd-profile-list.tsx`:

```typescript
"use client";

import type { ReactElement } from "react";
import { useState } from "react";
import { ActionIcon, Button, Group, Modal, Stack, Table, Text } from "@mantine/core";
import { Pencil, Trash2 } from "lucide-react";
import { HerdProfileForm } from "./herd-profile-form";
import {
  useCreateHerdProfile,
  useDeleteHerdProfile,
  useHerdProfiles,
  useUpdateHerdProfile,
} from "../hooks/use-herd-profiles";
import type { HerdProfile, HerdProfileCreate } from "@/types/api";

export function HerdProfileList(): ReactElement {
  const { data: profiles = [], isLoading } = useHerdProfiles();
  const createMutation = useCreateHerdProfile();
  const updateMutation = useUpdateHerdProfile();
  const deleteMutation = useDeleteHerdProfile();

  const [createOpen, setCreateOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<HerdProfile | null>(null);

  function handleCreate(data: HerdProfileCreate) {
    createMutation.mutate(data, { onSuccess: () => setCreateOpen(false) });
  }

  function handleUpdate(data: HerdProfileCreate) {
    if (!editTarget) return;
    updateMutation.mutate({ id: editTarget.id, data }, { onSuccess: () => setEditTarget(null) });
  }

  function handleDelete(profile: HerdProfile) {
    if (confirm(`Delete profile "${profile.name}"?`)) {
      deleteMutation.mutate(profile.id);
    }
  }

  if (isLoading) return <Text c="dimmed">Loading profiles…</Text>;

  return (
    <>
      <Stack gap="md">
        <Group justify="space-between">
          <Text fw={500}>Saved Herd Profiles</Text>
          <Button size="sm" color="violet" onClick={() => setCreateOpen(true)}>
            New profile
          </Button>
        </Group>

        {profiles.length === 0 ? (
          <Text c="dimmed" size="sm">
            No profiles yet. Create one to save a set of herd statistics.
          </Text>
        ) : (
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Description</Table.Th>
                <Table.Th>Created</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {profiles.map((profile) => (
                <Table.Tr key={profile.id}>
                  <Table.Td>{profile.name}</Table.Td>
                  <Table.Td c="dimmed">{profile.description || "—"}</Table.Td>
                  <Table.Td>
                    {profile.created_at
                      ? new Date(profile.created_at).toLocaleDateString()
                      : "—"}
                  </Table.Td>
                  <Table.Td>
                    <Group gap="xs" justify="flex-end">
                      <ActionIcon
                        variant="subtle"
                        onClick={() => setEditTarget(profile)}
                        aria-label="Edit profile"
                      >
                        <Pencil size={14} />
                      </ActionIcon>
                      <ActionIcon
                        variant="subtle"
                        color="red"
                        onClick={() => handleDelete(profile)}
                        aria-label="Delete profile"
                        loading={deleteMutation.isPending}
                      >
                        <Trash2 size={14} />
                      </ActionIcon>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        )}
      </Stack>

      <Modal opened={createOpen} onClose={() => setCreateOpen(false)} title="New herd profile" size="xl">
        <HerdProfileForm
          onSubmit={handleCreate}
          onCancel={() => setCreateOpen(false)}
          isLoading={createMutation.isPending}
        />
      </Modal>

      <Modal
        opened={editTarget !== null}
        onClose={() => setEditTarget(null)}
        title="Edit herd profile"
        size="xl"
      >
        {editTarget && (
          <HerdProfileForm
            initial={editTarget}
            onSubmit={handleUpdate}
            onCancel={() => setEditTarget(null)}
            isLoading={updateMutation.isPending}
          />
        )}
      </Modal>
    </>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd apps/frontend/dashboard && bun run tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add apps/frontend/dashboard/src/app/(dashboard)/herd-stats/components/herd-profile-list.tsx
git commit -m "feat: add HerdProfileList component with create/edit/delete modals"
```

---

### Task 10: Frontend — Herd Stats page + navigation

**Files:**
- Create: `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/page.tsx`
- Modify: `apps/frontend/dashboard/src/components/dashboard/navigation.ts`

- [ ] **Step 1: Create the page**

Create `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/page.tsx`:

```typescript
import type { ReactElement } from "react";
import { HerdProfileList } from "./components/herd-profile-list";

export default function HerdStatsPage(): ReactElement {
  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold">Herd Stats</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Manage saved herd profiles. Select a profile on the Curves page to use its
          statistics as a starting point for autoencoder predictions.
        </p>
      </div>
      <HerdProfileList />
    </div>
  );
}
```

- [ ] **Step 2: Add navigation entry**

Open `apps/frontend/dashboard/src/components/dashboard/navigation.ts`. Replace the full file contents with:

```typescript
import type { LucideIcon } from "lucide-react";
import { BarChart3, FlaskConical, Upload } from "lucide-react";

export interface NavigationItem {
  readonly label: string;
  readonly href: string;
  readonly icon: LucideIcon;
}

export const DASHBOARD_NAVIGATION: readonly NavigationItem[] = [
  { label: "Curves", href: "/curves", icon: FlaskConical },
  { label: "Herd Stats", href: "/herd-stats", icon: BarChart3 },
  { label: "Playground", href: "/playground", icon: Upload },
];
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd apps/frontend/dashboard && bun run tsc --noEmit
```

- [ ] **Step 4: Commit**

```bash
git add apps/frontend/dashboard/src/app/(dashboard)/herd-stats/page.tsx \
        apps/frontend/dashboard/src/components/dashboard/navigation.ts
git commit -m "feat: add herd stats page and navigation entry"
```

---

### Task 11: Frontend — Profile dropdown on curves page

**Files:**
- Modify: `apps/frontend/dashboard/src/app/(dashboard)/curves/components/autoencoder-input-panel.tsx`

- [ ] **Step 1: Add profile selector**

Open `apps/frontend/dashboard/src/app/(dashboard)/curves/components/autoencoder-input-panel.tsx`.

The file already imports `Select` on line 3. Do not add a duplicate import.

Add these two imports after the existing import statements:

```typescript
import { useHerdProfiles } from "@/app/(dashboard)/herd-stats/hooks/use-herd-profiles";
import { herdProfileToStats } from "@/lib/herd-profile-utils";
```

Inside the `AutoencoderInputPanel` function body, add after the `useDisclosure` line:

```typescript
  const { data: profiles = [] } = useHerdProfiles();
  const profileOptions = [
    { value: "", label: "None (manual)" },
    ...profiles.map((p) => ({ value: String(p.id), label: p.name })),
  ];
```

In the JSX, inside `<div className="space-y-3">`, add **before** the existing `<NumberInput label="Parity"` block:

```tsx
          <Select
            label="Herd profile preset"
            data={profileOptions}
            defaultValue=""
            onChange={(val) => {
              if (!val) return;
              const profile = profiles.find((p) => String(p.id) === val);
              if (profile) onHerdStatsChange(herdProfileToStats(profile));
            }}
            size="sm"
            placeholder="Select a saved profile…"
            clearable
          />
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd apps/frontend/dashboard && bun run tsc --noEmit
```

- [ ] **Step 3: Smoke test manually**

In terminal 1: `cd apps/backend/api && uv run python -m uvicorn bovi_api.app:app --reload --port 8000`
In terminal 2: `cd apps/frontend/dashboard && bun dev`

Open `http://localhost:3000/herd-stats`. Create a profile. Navigate to `/curves`, expand Herd Statistics, verify dropdown shows the saved profile, select it, verify sliders update.

- [ ] **Step 4: Commit**

```bash
git add apps/frontend/dashboard/src/app/(dashboard)/curves/components/autoencoder-input-panel.tsx
git commit -m "feat: add herd profile preset dropdown to autoencoder input panel"
```

---

## Phase 2 — CSV Upload

---

### Task 12: `HerdStatsRangeNormalizationTransform`

**Files:**
- Modify: `packages/models/lactation-autoencoder/src/lactation_autoencoder/dataloaders/transforms/lactation_transforms.py`
- Modify: `packages/models/lactation-autoencoder/data/experiments/lactation_autoencoder/versions/v15/config/config.yaml`
- Modify: `packages/models/lactation-autoencoder/tests/test_lactation_transforms.py`

- [ ] **Step 1: Write failing tests**

Open `packages/models/lactation-autoencoder/tests/test_lactation_transforms.py`. Add:

```python
def test_herd_stats_range_normalization_midpoint():
    """Value at range midpoint normalizes to 0.5."""
    from lactation_autoencoder.dataloaders.transforms.lactation_transforms import (
        HerdStatsRangeNormalizationTransform,
    )
    ranges = {
        "Achieved21Milk": (0.0, 50.0),
        "Achieved305Milk": (3000.0, 15000.0),
        "Achieved75Milk": (0.0, 50.0),
        "AchievedMilk": (3000.0, 20000.0),
        "DaysDry": (0.0, 150.0),
        "DaysInMilk": (0.0, 600.0),
        "DaysOpen": (0.0, 300.0),
        "DaysPregnant": (0.0, 283.0),
        "HistoricCalvingInterval": (300.0, 600.0),
        "QualitySequence": (0.0, 1.0),
    }
    transform = HerdStatsRangeNormalizationTransform(stat_ranges=ranges)
    raw = {
        "Achieved21Milk": 25.0,
        "Achieved305Milk": 9000.0,
        "Achieved75Milk": 25.0,
        "AchievedMilk": 11500.0,
        "DaysDry": 75.0,
        "DaysInMilk": 300.0,
        "DaysOpen": 150.0,
        "DaysPregnant": 141.5,
        "HistoricCalvingInterval": 450.0,
        "QualitySequence": 0.5,
    }
    result = transform({"herd_stats_raw": raw})
    for name, value in result["herd_stats_normalized"].items():
        assert abs(value - 0.5) < 1e-6, f"{name}: expected 0.5, got {value}"


def test_herd_stats_range_normalization_clamps():
    """Values outside range are clamped to [0, 1]."""
    from lactation_autoencoder.dataloaders.transforms.lactation_transforms import (
        HerdStatsRangeNormalizationTransform,
    )
    ranges = {k: (0.0, 1.0) for k in [
        "Achieved21Milk", "Achieved305Milk", "Achieved75Milk", "AchievedMilk",
        "DaysDry", "DaysInMilk", "DaysOpen", "DaysPregnant",
        "HistoricCalvingInterval", "QualitySequence",
    ]}
    transform = HerdStatsRangeNormalizationTransform(stat_ranges=ranges)
    raw = {k: 999.0 for k in ranges}
    result = transform({"herd_stats_raw": raw})
    for value in result["herd_stats_normalized"].values():
        assert value == 1.0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd packages/models/lactation-autoencoder && uv run pytest tests/test_lactation_transforms.py -k "range_normalization" -v
```

Expected: `ImportError` or `AttributeError`.

- [ ] **Step 3: Add `HerdStatsRangeNormalizationTransform` to `lactation_transforms.py`**

Open the file and add at the very end:

```python
class HerdStatsRangeNormalizationTransform(UniversalTransform):
    """Convert raw herd stat values (domain units) to 0–1 using fixed domain ranges.

    Not registered in TransformRegistry — used directly by the ingestion utility.
    Do not confuse with HerdStatsNormalizationTransform which applies per-sample
    z-score/minmax to already-normalized 0–1 values within a batch.

    Args:
        stat_ranges: Dict mapping canonical stat name to (min, max) raw domain range.

    """

    CANONICAL_ORDER: list[str] = [
        "Achieved21Milk",
        "Achieved305Milk",
        "Achieved75Milk",
        "AchievedMilk",
        "DaysDry",
        "DaysInMilk",
        "DaysOpen",
        "DaysPregnant",
        "HistoricCalvingInterval",
        "QualitySequence",
    ]

    def __init__(self, stat_ranges: dict[str, tuple[float, float]]) -> None:
        self.stat_ranges = stat_ranges

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """Normalize raw herd stats dict to 0–1.

        Args:
            data: Must contain ``"herd_stats_raw": dict[str, float]``.

        Returns:
            data with ``"herd_stats_normalized": dict[str, float]`` added.

        """
        raw: dict[str, float] = data["herd_stats_raw"]  # type: ignore[assignment]
        normalized: dict[str, float] = {}
        for name in self.CANONICAL_ORDER:
            value = float(raw[name])
            lo, hi = self.stat_ranges[name]
            clipped = max(lo, min(hi, value))
            normalized[name] = (clipped - lo) / (hi - lo) if hi > lo else 0.0
        data["herd_stats_normalized"] = normalized
        return data

    def get_params(self) -> dict[str, object]:
        """Return parameters for reproducibility."""
        return {"stat_ranges": self.stat_ranges}
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd packages/models/lactation-autoencoder && uv run pytest tests/test_lactation_transforms.py -k "range_normalization" -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Add `herd_stats_ranges` to config.yaml**

Open `packages/models/lactation-autoencoder/data/experiments/lactation_autoencoder/versions/v15/config/config.yaml`. Add at the end:

```yaml
herd_stats_ranges:
  Achieved21Milk:          [0.0, 50.0]
  Achieved305Milk:         [3000.0, 15000.0]
  Achieved75Milk:          [0.0, 50.0]
  AchievedMilk:            [3000.0, 20000.0]
  DaysDry:                 [0.0, 150.0]
  DaysInMilk:              [0.0, 600.0]
  DaysOpen:                [0.0, 300.0]
  DaysPregnant:            [0.0, 283.0]
  HistoricCalvingInterval: [300.0, 600.0]
  QualitySequence:         [0.0, 1.0]
```

- [ ] **Step 6: Commit**

```bash
git add packages/models/lactation-autoencoder/src/lactation_autoencoder/dataloaders/transforms/lactation_transforms.py \
        packages/models/lactation-autoencoder/tests/test_lactation_transforms.py \
        packages/models/lactation-autoencoder/data/experiments/lactation_autoencoder/versions/v15/config/config.yaml
git commit -m "feat: add HerdStatsRangeNormalizationTransform and herd_stats_ranges config"
```

---

### Task 13: Ingestion utility

**Files:**
- Create: `apps/backend/api/tests/test_herd_stats_ingestion.py`
- Create: `apps/backend/api/src/bovi_api/herd_stats_ingestion.py`

- [ ] **Step 1: Write failing tests first**

Create `apps/backend/api/tests/test_herd_stats_ingestion.py`:

```python
"""Unit tests for the herd stats CSV ingestion utility."""

import pytest
from bovi_api.herd_stats_ingestion import parse_csv, normalize_herd_stats

FULL_HEADER = "Achieved21Milk,Achieved305Milk,Achieved75Milk,AchievedMilk,DaysDry,DaysInMilk,DaysOpen,DaysPregnant,HistoricCalvingInterval,QualitySequence"
FULL_ROW = "25.0,9000.0,28.0,10000.0,60.0,180.0,100.0,150.0,420.0,0.8"
AGGREGATED_CSV = f"{FULL_HEADER}\n{FULL_ROW}\n".encode()
INDIVIDUAL_CSV = f"{FULL_HEADER}\n20.0,8000.0,25.0,9000.0,55.0,160.0,90.0,140.0,400.0,0.7\n30.0,10000.0,31.0,11000.0,65.0,200.0,110.0,160.0,440.0,0.9\n".encode()


def test_parse_aggregated_csv():
    raw, fmt, row_count, warnings = parse_csv(AGGREGATED_CSV)
    assert fmt == "aggregated"
    assert row_count == 1
    assert raw["Achieved21Milk"] == pytest.approx(25.0)
    assert warnings == []


def test_parse_individual_csv_computes_mean():
    raw, fmt, row_count, warnings = parse_csv(INDIVIDUAL_CSV)
    assert fmt == "individual"
    assert row_count == 2
    assert raw["Achieved21Milk"] == pytest.approx(25.0)   # mean of 20 and 30
    assert raw["Achieved305Milk"] == pytest.approx(9000.0) # mean of 8000 and 10000


def test_parse_partial_columns_returns_warning():
    partial = b"Achieved21Milk,Achieved305Milk\n25.0,9000.0\n"
    raw, _, _, warnings = parse_csv(partial)
    assert "Achieved21Milk" in raw
    assert len(warnings) > 0


def test_parse_no_recognised_columns_raises():
    with pytest.raises(ValueError, match="No recognised herd stat columns"):
        parse_csv(b"cow_id,breed\n1,Holstein\n")


def test_parse_unparseable_raises():
    with pytest.raises(ValueError, match="Not a valid CSV"):
        parse_csv(b"\x00\x01\x02\x03")


def test_parse_alias_dim_maps_to_days_in_milk():
    alias_csv = f"DIM,Achieved305Milk,Achieved21Milk,Achieved75Milk,AchievedMilk,DaysDry,DaysOpen,DaysPregnant,HistoricCalvingInterval,QualitySequence\n180.0,9000.0,25.0,28.0,10000.0,60.0,100.0,150.0,420.0,0.8\n".encode()
    raw, _, _, _ = parse_csv(alias_csv)
    assert "DaysInMilk" in raw
    assert raw["DaysInMilk"] == pytest.approx(180.0)


def test_parse_row_cap_truncates_with_warning():
    header = f"{FULL_HEADER}\n".encode()
    row = f"{FULL_ROW}\n".encode()
    big_csv = header + row * 5
    _, _, row_count, warnings = parse_csv(big_csv, max_rows=3)
    assert row_count == 3
    assert any("3" in w for w in warnings)


def test_normalize_herd_stats_midpoint():
    ranges = {k: (0.0, 100.0) for k in ["Achieved21Milk", "Achieved305Milk",
        "Achieved75Milk", "AchievedMilk", "DaysDry", "DaysInMilk",
        "DaysOpen", "DaysPregnant", "HistoricCalvingInterval", "QualitySequence"]}
    raw = {k: 50.0 for k in ranges}
    result = normalize_herd_stats(raw, ranges)
    for val in result.values():
        assert val == pytest.approx(0.5)


def test_normalize_herd_stats_clamps():
    ranges = {k: (0.0, 1.0) for k in ["Achieved21Milk", "Achieved305Milk",
        "Achieved75Milk", "AchievedMilk", "DaysDry", "DaysInMilk",
        "DaysOpen", "DaysPregnant", "HistoricCalvingInterval", "QualitySequence"]}
    raw = {k: 999.0 for k in ranges}
    result = normalize_herd_stats(raw, ranges)
    for val in result.values():
        assert val == 1.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd apps/backend/api && uv run pytest tests/test_herd_stats_ingestion.py -v
```

Expected: `ModuleNotFoundError` (module doesn't exist yet).

- [ ] **Step 3: Create `herd_stats_ingestion.py`**

Create `apps/backend/api/src/bovi_api/herd_stats_ingestion.py`:

```python
"""CSV ingestion and normalization utility for herd stats upload.

Two public functions:
  - parse_csv: reads CSV bytes → raw dict of domain values
  - normalize_herd_stats: maps raw domain values → 0-1 using config ranges

Normalization is inlined here (not imported from lactation_autoencoder) to keep
the central API free of ML framework dependencies.
"""

import csv
import io
from typing import Literal

CANONICAL_NAMES: list[str] = [
    "Achieved21Milk",
    "Achieved305Milk",
    "Achieved75Milk",
    "AchievedMilk",
    "DaysDry",
    "DaysInMilk",
    "DaysOpen",
    "DaysPregnant",
    "HistoricCalvingInterval",
    "QualitySequence",
]

# Case-insensitive column header → canonical stat name
_ALIASES: dict[str, str] = {
    name.lower(): name for name in CANONICAL_NAMES
}
_ALIASES.update({
    "dim": "DaysInMilk",
    "days_in_milk": "DaysInMilk",
    "daysinmilk": "DaysInMilk",
    "305milk": "Achieved305Milk",
    "21milk": "Achieved21Milk",
    "75milk": "Achieved75Milk",
    "total_milk": "AchievedMilk",
    "totalmilk": "AchievedMilk",
    "calvinginterval": "HistoricCalvingInterval",
    "calving_interval": "HistoricCalvingInterval",
    "qualitysequence": "QualitySequence",
    "quality": "QualitySequence",
    "daysdry": "DaysDry",
    "days_dry": "DaysDry",
    "daysopen": "DaysOpen",
    "days_open": "DaysOpen",
    "dayspregnant": "DaysPregnant",
    "days_pregnant": "DaysPregnant",
})


def _resolve_column(header: str) -> str | None:
    """Map a CSV column header to a canonical stat name, or None if unrecognised."""
    return _ALIASES.get(header.strip().lower())


def parse_csv(
    content: bytes,
    max_rows: int = 100_000,
) -> tuple[dict[str, float], Literal["aggregated", "individual"], int, list[str]]:
    """Parse uploaded CSV bytes into raw herd stats.

    Args:
        content: Raw CSV bytes from the uploaded file.
        max_rows: Maximum rows to process; excess rows are truncated with a warning.

    Returns:
        (raw_stats, format_detected, row_count, warnings)

    Raises:
        ValueError: If bytes are not parseable as CSV, or no recognised columns are found.

    """
    try:
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
    except Exception as exc:
        raise ValueError("Not a valid CSV file") from exc

    if not rows:
        raise ValueError("Not a valid CSV file — no data rows found")

    # Map headers to canonical names
    col_map: dict[str, str] = {}  # original header → canonical name
    for header in rows[0]:
        canonical = _resolve_column(header)
        if canonical is not None:
            col_map[header] = canonical

    if not col_map:
        expected = ", ".join(CANONICAL_NAMES[:5]) + ", ..."
        raise ValueError(
            f"No recognised herd stat columns found. Expected one or more of: {expected}"
        )

    warnings: list[str] = []

    # Report missing canonical columns
    found_canonical = set(col_map.values())
    missing = [n for n in CANONICAL_NAMES if n not in found_canonical]
    if missing:
        warnings.append(f"Missing columns (absent from result): {', '.join(missing)}")

    # Truncate if over max_rows
    if len(rows) > max_rows:
        rows = rows[:max_rows]
        warnings.append(
            f"File had more rows than the limit of {max_rows}; only the first {max_rows} were used."
        )

    row_count = len(rows)
    format_detected: Literal["aggregated", "individual"] = (
        "aggregated" if row_count == 1 else "individual"
    )

    # Collect values per canonical column
    column_values: dict[str, list[float]] = {c: [] for c in col_map.values()}
    nan_counts: dict[str, int] = {c: 0 for c in col_map.values()}

    for row in rows:
        for header, canonical in col_map.items():
            raw_val = (row.get(header) or "").strip()
            try:
                column_values[canonical].append(float(raw_val))
            except (ValueError, TypeError):
                nan_counts[canonical] += 1

    for canonical, count in nan_counts.items():
        if count > 0:
            warnings.append(
                f"Column '{canonical}' had {count} unparseable value(s); excluded from mean."
            )

    raw_stats: dict[str, float] = {
        canonical: sum(values) / len(values)
        for canonical, values in column_values.items()
        if values
    }

    return raw_stats, format_detected, row_count, warnings


def normalize_herd_stats(
    raw: dict[str, float],
    stat_ranges: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Map raw domain values to 0–1 using provided ranges.

    Values outside the range are clamped. Only keys present in both raw and
    stat_ranges are normalized; missing keys are omitted from the output.

    Args:
        raw: Dict of canonical stat name → raw domain value.
        stat_ranges: Dict of canonical stat name → (min, max) range.

    Returns:
        Dict of canonical stat name → normalized float in [0, 1].

    """
    result: dict[str, float] = {}
    for name, (lo, hi) in stat_ranges.items():
        if name not in raw:
            continue
        value = float(raw[name])
        clipped = max(lo, min(hi, value))
        result[name] = (clipped - lo) / (hi - lo) if hi > lo else 0.0
    return result
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
cd apps/backend/api && uv run pytest tests/test_herd_stats_ingestion.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/backend/api/src/bovi_api/herd_stats_ingestion.py \
        apps/backend/api/tests/test_herd_stats_ingestion.py
git commit -m "feat: add herd stats CSV ingestion utility with tests"
```

---

### Task 14: CSV preview endpoint

**Files:**
- Modify: `apps/backend/api/src/bovi_api/routes/herd_profiles.py`
- Modify: `apps/backend/api/pyproject.toml`

- [ ] **Step 1: Add test for csv-preview endpoint**

Append to `apps/backend/api/tests/test_herd_profiles.py`:

```python
SAMPLE_CSV = (
    b"Achieved21Milk,Achieved305Milk,Achieved75Milk,AchievedMilk,"
    b"DaysDry,DaysInMilk,DaysOpen,DaysPregnant,HistoricCalvingInterval,QualitySequence\n"
    b"25.0,9000.0,28.0,10000.0,60.0,180.0,100.0,150.0,420.0,0.8\n"
)


def test_csv_preview_returns_normalized_stats(client):
    response = client.post(
        "/herd-profiles/csv-preview",
        files={"file": ("herd.csv", SAMPLE_CSV, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert data["format_detected"] == "aggregated"
    assert data["row_count"] == 1
    for value in data["stats"].values():
        assert 0.0 <= value <= 1.0


def test_csv_preview_rejects_non_csv_extension(client):
    response = client.post(
        "/herd-profiles/csv-preview",
        files={"file": ("data.xlsx", b"PK\x03\x04", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_csv_preview_rejects_unrecognised_columns(client):
    response = client.post(
        "/herd-profiles/csv-preview",
        files={"file": ("bad.csv", b"breed,farm\nHolstein,Farm1\n", "text/csv")},
    )
    assert response.status_code == 400
```

- [ ] **Step 2: Run tests — verify new tests fail**

```bash
cd apps/backend/api && uv run pytest tests/test_herd_profiles.py -k "csv_preview" -v
```

Expected: `404` (endpoint doesn't exist yet).

- [ ] **Step 3: Add `python-multipart` dependency**

Open `apps/backend/api/pyproject.toml`. Add to `dependencies`:

```toml
    "python-multipart>=0.0.12",
```

Run:
```bash
cd apps/backend/api && uv sync
```

- [ ] **Step 4: Add the csv-preview endpoint to `herd_profiles.py`**

Add imports at the top of `apps/backend/api/src/bovi_api/routes/herd_profiles.py`:

```python
from fastapi import File, UploadFile
from pydantic import BaseModel

from bovi_api.herd_stats_ingestion import normalize_herd_stats, parse_csv
```

Add the default ranges constant and response model at module level (after imports, before `router = APIRouter(...)`):

```python
# Biological estimate ranges for each stat (calibrate from training data in Phase 3)
_DEFAULT_STAT_RANGES: dict[str, tuple[float, float]] = {
    "Achieved21Milk": (0.0, 50.0),
    "Achieved305Milk": (3000.0, 15000.0),
    "Achieved75Milk": (0.0, 50.0),
    "AchievedMilk": (3000.0, 20000.0),
    "DaysDry": (0.0, 150.0),
    "DaysInMilk": (0.0, 600.0),
    "DaysOpen": (0.0, 300.0),
    "DaysPregnant": (0.0, 283.0),
    "HistoricCalvingInterval": (300.0, 600.0),
    "QualitySequence": (0.0, 1.0),
}

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class HerdProfileUploadResponse(BaseModel):
    """Preview of normalized herd stats parsed from a CSV upload. Not saved to DB."""

    stats: dict[str, float]
    format_detected: str
    row_count: int
    warnings: list[str]
```

Add the endpoint. It must be inserted **before** `@router.get("/{profile_id}")` so the static path `/csv-preview` is registered first:

```python
@router.post("/csv-preview", response_model=HerdProfileUploadResponse)
async def csv_preview(
    file: UploadFile = File(...),
) -> HerdProfileUploadResponse:
    """Parse and normalize an uploaded CSV. Returns a preview; does NOT save to DB."""
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")

    try:
        raw_stats, format_detected, row_count, warnings = parse_csv(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized = normalize_herd_stats(raw_stats, _DEFAULT_STAT_RANGES)

    return HerdProfileUploadResponse(
        stats=normalized,
        format_detected=format_detected,
        row_count=row_count,
        warnings=warnings,
    )
```

- [ ] **Step 5: Run all backend tests — verify they all pass**

```bash
cd apps/backend/api && uv run pytest tests/ -v
```

Expected: all 12 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/backend/api/src/bovi_api/routes/herd_profiles.py \
        apps/backend/api/pyproject.toml \
        apps/backend/api/tests/test_herd_profiles.py
git commit -m "feat: add csv-preview endpoint for herd stats upload"
```

---

### Task 15: Frontend — CSV upload component

**Files:**
- Modify: `apps/frontend/dashboard/src/types/api.ts`
- Modify: `apps/frontend/dashboard/src/lib/api-client.ts`
- Create: `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/hooks/use-herd-profile-upload.ts`
- Create: `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/components/herd-profile-upload.tsx`
- Modify: `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/page.tsx`

- [ ] **Step 1: Add upload response type to `types/api.ts`**

Append to `apps/frontend/dashboard/src/types/api.ts`:

```typescript
export const HerdProfileUploadResponseSchema = z.object({
  stats: z.record(z.string(), z.number()),
  format_detected: z.enum(["aggregated", "individual"]),
  row_count: z.number(),
  warnings: z.array(z.string()),
});

export type HerdProfileUploadResponse = z.infer<typeof HerdProfileUploadResponseSchema>;
```

- [ ] **Step 2: Add `uploadHerdProfileCsv` to `api-client.ts`**

Add to the imports at the top:

```typescript
import type { HerdProfileUploadResponse } from "@/types/api";
import { HerdProfileUploadResponseSchema } from "@/types/api";
```

Add function after `deleteHerdProfile`:

```typescript
export async function uploadHerdProfileCsv(
  file: File
): Promise<HerdProfileUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${getApiBaseUrl()}/herd-profiles/csv-preview`, {
    method: "POST",
    body: formData,
    // No Content-Type header — browser sets multipart boundary automatically
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`Upload error ${response.status}: ${JSON.stringify(error)}`);
  }
  const data: unknown = await response.json();
  return HerdProfileUploadResponseSchema.parse(data);
}
```

- [ ] **Step 3: Create the upload hook**

Create `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/hooks/use-herd-profile-upload.ts`:

```typescript
"use client";

import { useMutation } from "@tanstack/react-query";
import { uploadHerdProfileCsv } from "@/lib/api-client";

export function useHerdProfileUpload() {
  return useMutation({
    mutationFn: (file: File) => uploadHerdProfileCsv(file),
  });
}
```

- [ ] **Step 4: Create the upload component**

Create `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/components/herd-profile-upload.tsx`:

```typescript
"use client";

import type { ReactElement } from "react";
import { useRef, useState } from "react";
import { Alert, Button, Divider, Group, Modal, Stack, Table, Text } from "@mantine/core";
import { AlertCircle } from "lucide-react";
import { HERD_STATS_METADATA } from "@/data/herd-stats-metadata";
import { statsToHerdProfileFields } from "@/lib/herd-profile-utils";
import type { HerdProfileUploadResponse } from "@/types/api";
import { HerdProfileForm } from "./herd-profile-form";
import { useHerdProfileUpload } from "../hooks/use-herd-profile-upload";
import { useCreateHerdProfile } from "../hooks/use-herd-profiles";

export function HerdProfileUpload(): ReactElement {
  const inputRef = useRef<HTMLInputElement>(null);
  const uploadMutation = useHerdProfileUpload();
  const createMutation = useCreateHerdProfile();
  const [preview, setPreview] = useState<HerdProfileUploadResponse | null>(null);
  const [saveOpen, setSaveOpen] = useState(false);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    uploadMutation.mutate(file, { onSuccess: setPreview });
    e.target.value = ""; // allow re-upload of same file
  }

  function getPreviewStatsArray(): number[] {
    if (!preview) return [];
    return HERD_STATS_METADATA.map((meta) => preview.stats[meta.name] ?? 0);
  }

  return (
    <>
      <Stack gap="sm">
        <Text size="sm" fw={500}>Import from CSV</Text>
        <Text size="xs" c="dimmed">
          Upload a CSV with herd stat columns. One-row CSVs are treated as aggregated;
          multi-row CSVs are averaged per column. Values are normalized to 0–1 automatically.
        </Text>

        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          style={{ display: "none" }}
          onChange={handleFileChange}
        />
        <Button
          variant="outline"
          size="sm"
          onClick={() => inputRef.current?.click()}
          loading={uploadMutation.isPending}
          style={{ width: "fit-content" }}
        >
          Choose CSV file…
        </Button>

        {uploadMutation.isError && (
          <Alert icon={<AlertCircle size={16} />} color="red" title="Upload failed">
            {(uploadMutation.error as Error).message}
          </Alert>
        )}

        {preview && (
          <Stack gap="sm">
            <Text size="xs" c="dimmed">
              Format: {preview.format_detected} · {preview.row_count} row(s) processed
            </Text>
            {preview.warnings.map((w, i) => (
              <Alert key={i} icon={<AlertCircle size={14} />} color="yellow" size="xs">
                {w}
              </Alert>
            ))}
            <Table striped compact withColumnBorders>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Stat</Table.Th>
                  <Table.Th>Normalized (0–1)</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {HERD_STATS_METADATA.map((meta) => (
                  <Table.Tr key={meta.name}>
                    <Table.Td>{meta.label}</Table.Td>
                    <Table.Td>{preview.stats[meta.name]?.toFixed(3) ?? "—"}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
            <Group>
              <Button size="sm" color="violet" onClick={() => setSaveOpen(true)}>
                Save as profile…
              </Button>
              <Button size="sm" variant="subtle" onClick={() => setPreview(null)}>
                Discard
              </Button>
            </Group>
          </Stack>
        )}
      </Stack>

      <Modal
        opened={saveOpen}
        onClose={() => setSaveOpen(false)}
        title="Save uploaded herd profile"
        size="xl"
      >
        <HerdProfileForm
          initial={
            preview
              ? ({
                  id: -1,
                  name: "",
                  description: "",
                  created_at: null,
                  updated_at: null,
                  ...statsToHerdProfileFields(getPreviewStatsArray()),
                } as any)
              : undefined
          }
          onSubmit={(data) => {
            createMutation.mutate(data, {
              onSuccess: () => {
                setSaveOpen(false);
                setPreview(null);
              },
            });
          }}
          onCancel={() => setSaveOpen(false)}
          isLoading={createMutation.isPending}
        />
      </Modal>
    </>
  );
}
```

- [ ] **Step 5: Wire upload into the page**

Open `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/page.tsx`. Replace with:

```typescript
import type { ReactElement } from "react";
import { Divider } from "@mantine/core";
import { HerdProfileList } from "./components/herd-profile-list";
import { HerdProfileUpload } from "./components/herd-profile-upload";

export default function HerdStatsPage(): ReactElement {
  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold">Herd Stats</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Manage saved herd profiles. Select a profile on the Curves page to use its
          statistics as a starting point for autoencoder predictions.
        </p>
      </div>
      <HerdProfileList />
      <Divider label="Import" labelPosition="left" />
      <HerdProfileUpload />
    </div>
  );
}
```

- [ ] **Step 6: Verify TypeScript compiles**

```bash
cd apps/frontend/dashboard && bun run tsc --noEmit
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add apps/frontend/dashboard/src/types/api.ts \
        apps/frontend/dashboard/src/lib/api-client.ts \
        apps/frontend/dashboard/src/app/(dashboard)/herd-stats/
git commit -m "feat: add CSV upload component and herd profile import flow"
```

---

### Task 16: End-to-end verification

- [ ] **Step 1: Run all backend tests**

```bash
cd apps/backend/api && uv run pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run lactation-autoencoder package tests**

```bash
cd packages/models/lactation-autoencoder && uv run pytest tests/ -v
```

Expected: all tests PASS including the 2 new range normalization tests.

- [ ] **Step 3: Manual smoke test — full stack**

In terminal 1:
```bash
cd apps/backend/api && uv run python -m uvicorn bovi_api.app:app --reload --port 8000
```

In terminal 2:
```bash
cd apps/frontend/dashboard && bun dev
```

Perform these checks in order:
1. Open `http://localhost:3000/herd-stats`
2. Create a profile manually (name + slider values) → verify it appears in the list
3. Edit the profile → verify changes persist
4. Navigate to `/curves`, expand Herd Statistics, verify dropdown lists the saved profile
5. Select profile → verify sliders update with profile values
6. Run a prediction → verify it completes without errors
7. Return to `/herd-stats`, upload an aggregated CSV → verify normalized preview
8. Upload an individual cow CSV (multiple rows) → verify mean aggregation in preview
9. Save from preview → verify profile appears in list
10. Delete a profile → verify it disappears

- [ ] **Step 4: Final commit**

```bash
git add apps/backend/api/src/bovi_api/routes/herd_profiles.py
git commit -m "feat: complete herd stats management (Phase 1 + Phase 2)"
```
