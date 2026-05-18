# Herd Stats Management — Design Spec

**Date:** 2026-04-15
**Status:** Reviewed

---

## Scope & dependencies

This spec covers herd stats management **without authentication**. Multi-user support (Azure AD + Entra ID for authentication, own `users` table for profile isolation) is a planned follow-up and will be a separate spec. The database schema here is designed to make that migration easy: the `user_id` foreign key is noted as a TODO but not implemented yet.

---

## Context

The lactation autoencoder model takes 10 herd-level statistics as input (normalized 0–1). Currently these come from pickle files via a 4-level fallback (herd+parity → herd → parity → global). The frontend curves page has sliders for on-the-fly adjustment (in `autoencoder-input-panel.tsx`), but there is no way to save, load, or upload custom herd profiles.

This feature adds:
1. A dedicated **Herd Stats management page** where users create, edit, and delete named herd profiles
2. **CSV upload** (pre-aggregated or individual cow data) that normalizes to 0–1 automatically
3. **Persistent storage** in the central API database (SQLite on Azure Files in Azure/prod; local SQLite in dev)
4. **Dropdown integration** on the curves page to select a saved profile as a starting preset (sliders remain editable)

---

## Architecture

### Principle: separate ingestion from normalization

CSV parsing is an ETL/ingestion operation (file → raw domain values). Range-based normalization is a stateless mapping (raw values → 0–1). They are split into two layers:

- **Ingestion utility** (`apps/backend/api/src/bovi_api/herd_stats_ingestion.py`) — parses CSV to a raw dict, validates structure, returns warnings. No ML machinery.
- **`HerdStatsRangeNormalizationTransform`** — a standalone class (following `UniversalTransform` interface but **not registered in `TransformRegistry`**) that maps raw domain values to 0–1 using config-defined ranges. Defined in the lactation-autoencoder package since ranges are model-specific. Can be promoted to a registered transform later if it is ever needed in a training pipeline.

---

## Components

### 1. Normalization class: `HerdStatsRangeNormalizationTransform`

**File:** `packages/models/lactation-autoencoder/src/lactation_autoencoder/dataloaders/transforms/lactation_transforms.py`

Not registered in `TransformRegistry` — it is a standalone utility class that follows the `UniversalTransform` interface for consistency. It is instantiated directly by the ingestion utility. It is not part of the inference pipeline (the existing `HerdStatsEnrichmentTransform` already handles that path via pickle lookup).

> Note: the existing `HerdStatsNormalizationTransform` (registered as `"herd_stats_normalization"`) operates on already-normalized 0–1 values (z-score/per-sample minmax within a batch). This new class converts **raw domain values** (e.g. 9500 kg for 305-day milk) **to** 0–1. They address different stages and should not be confused.

```python
class HerdStatsRangeNormalizationTransform(UniversalTransform):
    """Convert raw herd stat values (domain units) to 0-1 using config-defined ranges."""

    CANONICAL_ORDER = [
        "Achieved21Milk", "Achieved305Milk", "Achieved75Milk", "AchievedMilk",
        "DaysDry", "DaysInMilk", "DaysOpen", "DaysPregnant",
        "HistoricCalvingInterval", "QualitySequence",
    ]

    def __init__(self, stat_ranges: dict[str, tuple[float, float]]) -> None:
        self.stat_ranges = stat_ranges

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        raw: dict[str, float] = data["herd_stats_raw"]  # set by caller
        normalized: dict[str, float] = {}
        for name in self.CANONICAL_ORDER:
            value = float(raw[name])
            lo, hi = self.stat_ranges[name]
            clipped = max(lo, min(hi, value))
            normalized[name] = (clipped - lo) / (hi - lo) if hi > lo else 0.0
        data["herd_stats_normalized"] = normalized
        return data

    def get_params(self) -> dict[str, object]:
        return {"stat_ranges": self.stat_ranges}
```

The caller (ingestion utility) constructs `{"herd_stats_raw": raw_dict}`, calls the transform, then reads `data["herd_stats_normalized"]` as a `dict[str, float]`.

**Config.yaml addition** (in `data/experiments/lactation_autoencoder/versions/v15/config/config.yaml`):

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

Ranges are initial biological estimates. Calibrate against actual training data statistics in Phase 3 — only config changes, no code changes needed.

---

### 2. Ingestion utility

**File:** `apps/backend/api/src/bovi_api/herd_stats_ingestion.py`

Co-located with the central API (the only consumer). Pure Python, no ML framework imports.

```python
COLUMN_ALIASES: dict[str, str] = {
    "dim": "DaysInMilk",
    "days_in_milk": "DaysInMilk",
    "305milk": "Achieved305Milk",
    "21milk": "Achieved21Milk",
    # ... more aliases (case-insensitive matching applied before lookup)
}

def parse_csv(
    content: bytes,
    max_rows: int = 100_000,
) -> tuple[dict[str, float], str, int, list[str]]:
    """
    Parse uploaded CSV bytes.

    Returns:
        (raw_stats, format_detected, row_count, warnings)
        - raw_stats: dict[canonical_name, float] — one entry per recognised column
        - format_detected: "aggregated" (1 data row) or "individual" (>1 rows, mean taken)
        - row_count: number of data rows parsed
        - warnings: list of human-readable warnings (missing columns, NaN values, etc.)

    Raises:
        ValueError: if the file is not parseable as CSV, or if no recognised columns are found.
    """
    ...
```

Format detection: 1 data row → "aggregated"; >1 rows → "individual" (column-wise mean, NaN rows excluded with a warning).

**Error contract:**
- `ValueError("Not a valid CSV file")` — unparseable bytes
- `ValueError("No recognised herd stat columns found. Expected one or more of: ...")` — zero matching columns
- Partial matches return `warnings` entries; the endpoint returns HTTP 200 with warnings so the user can decide whether to proceed

File constraints (enforced by the upload endpoint, not this utility):
- Maximum file size: 10 MB
- Accepted content types: `text/csv`, `text/plain`, `application/octet-stream` (extension check: `.csv`)
- Row cap: 100,000 rows — if exceeded, truncate to first 100,000 rows and add a warning `"File had N rows, only the first 100,000 were used"`. Do not error; truncation is preferable to rejecting a large file that would otherwise produce valid stats.

---

### 3. Database

**File:** `apps/backend/api/src/bovi_api/models.py`

New `HerdProfile` SQLModel table alongside existing `FittingResult`.

```python
from sqlalchemy import Column, DateTime, func, UniqueConstraint

class HerdProfileBase(SQLModel):
    name: str = Field(max_length=100)
    description: str = Field(default="", max_length=500)
    # 10 stats stored as individual typed columns (fixed schema, DB-level 0-1 constraint)
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
    __tablename__ = "herd_profiles"
    __table_args__ = (UniqueConstraint("name", name="uq_herd_profile_name"),)

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

class HerdProfileCreate(HerdProfileBase): ...
class HerdProfileRead(HerdProfileBase):
    id: int
    created_at: datetime
    updated_at: datetime
```

`name` has a DB-level unique constraint for now (global uniqueness). When multi-user support is added (Azure AD + own `users` table), replace `UniqueConstraint("name")` with a `UniqueConstraint("user_id", "name")` compound index and add a `user_id: int = Field(foreign_key="users.id", index=True)` column. A `PUT` to an existing profile with a name collision returns HTTP 409.

Requires an Alembic migration (`alembic revision --autogenerate -m "add herd_profiles table"`). Do not use `create_tables()` for production — Alembic is the migration tool. Migration files live in `apps/backend/api/alembic/versions/`.

**`updated_at` strategy:** `server_default=func.now()` sets the value on INSERT. For auto-update on UPDATE, use SQLAlchemy `onupdate=func.now()` or a SQLite-compatible migration. If this feature is deployed against an explicitly configured PostgreSQL database, a PostgreSQL trigger would look like:

```sql
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_herd_profiles_updated_at
    BEFORE UPDATE ON herd_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

This is server-side (works for ORM updates, raw SQL updates, and migrations alike), unlike the client-side `onupdate` SQLAlchemy hook which only fires through ORM sessions.

---

### 4. API endpoints

**New file:** `apps/backend/api/src/bovi_api/routes/herd_profiles.py`

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/herd-profiles/` | List all profiles |
| `POST` | `/herd-profiles/` | Create profile (JSON body) |
| `GET` | `/herd-profiles/{id}` | Get single profile |
| `PUT` | `/herd-profiles/{id}` | Update profile |
| `DELETE` | `/herd-profiles/{id}` | Delete profile (returns 204) |
| `POST` | `/herd-profiles/csv-preview` | Parse + normalize CSV, return preview (does NOT save) |

> **Route ordering note:** `csv-preview` is a static path segment and must be registered **before** the `{id}` routes in the router to avoid FastAPI matching `"csv-preview"` as an `id` parameter. In practice, FastAPI resolves static segments before parameterized ones if registered first.

The upload endpoint is **stateless** — parses, normalizes, returns a preview. The user confirms by calling `POST /herd-profiles/` separately with the (optionally adjusted) values.

**Upload response** — uses named fields, not a positional list:

```python
class HerdProfileUploadResponse(BaseModel):
    stats: dict[str, float]     # {"Achieved21Milk": 0.53, ...} — canonical names
    format_detected: str        # "aggregated" or "individual"
    row_count: int
    warnings: list[str]         # e.g. "Column 'DaysOpen' had 3 missing values, filled with column mean"
```

Using `dict[str, float]` (not `list[float]`) avoids positional coupling between backend and frontend: the frontend maps by key name to the corresponding `HerdProfileCreate` field.

**Error responses:**
- `400` with `{"detail": "Not a valid CSV file"}` — unparseable bytes
- `400` with `{"detail": "No recognised herd stat columns found. ..."}` — zero matching columns
- `413` — file exceeds 10 MB
- `409` on `POST /herd-profiles/` — profile name already exists

Register router in `apps/backend/api/src/bovi_api/app.py` with `app.include_router(herd_profiles.router, prefix="/herd-profiles")`.

---

### 5. Frontend: Herd Stats management page

**New directory:** `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/`

```
herd-stats/
  page.tsx                        # Page shell
  components/
    herd-profile-list.tsx         # Table with edit/delete actions
    herd-profile-form.tsx         # Create/edit form (reuses HerdStatsForm sliders + name/description)
    herd-profile-upload.tsx       # CSV drop zone + preview of normalized values
  hooks/
    use-herd-profiles.ts          # React Query: list, create, update, delete
    use-herd-profile-upload.ts    # React Query mutation for CSV upload
```

`herd-profile-form.tsx` reuses the existing `HerdStatsForm` slider component from `autoencoder/components/herd-stats-form.tsx` plus name and description text fields.

**Canonical field mapping** — the frontend needs a stable mapping between the snake_case DB field names and the PascalCase stat names used in `HERD_STATS_METADATA` and `HerdStatsForm` (`values: number[]`). Define this constant in `apps/frontend/dashboard/src/data/herd-stats-metadata.ts` (or a new `herd-profile-utils.ts`):

```typescript
// Maps HerdProfile DB field names → HerdStatsMetadata index order
// Must match HERD_STATS_METADATA stat name order exactly
export const HERD_PROFILE_FIELD_ORDER: (keyof HerdProfileCreate)[] = [
  "achieved_21_milk",       // index 0: Achieved21Milk
  "achieved_305_milk",      // index 1: Achieved305Milk
  "achieved_75_milk",       // index 2: Achieved75Milk
  "achieved_milk",          // index 3: AchievedMilk
  "days_dry",               // index 4: DaysDry
  "days_in_milk",           // index 5: DaysInMilk
  "days_open",              // index 6: DaysOpen
  "days_pregnant",          // index 7: DaysPregnant
  "historic_calving_interval", // index 8: HistoricCalvingInterval
  "quality_sequence",       // index 9: QualitySequence
]

export function herdProfileToStats(profile: HerdProfile): number[] {
  return HERD_PROFILE_FIELD_ORDER.map(field => profile[field] as number)
}
```

When a profile is selected in the dropdown, call `onHerdStatsChange(herdProfileToStats(selectedProfile))`.

**New Zod types** in `apps/frontend/dashboard/src/types/api.ts`:

```typescript
const HerdProfileSchema = z.object({
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
  created_at: z.string(),
  updated_at: z.string(),
})

const HerdProfileCreateSchema = HerdProfileSchema.omit({ id: true, created_at: true, updated_at: true })
// Inherits max(100)/max(500) from HerdProfileSchema for name/description

const HerdProfileUploadResponseSchema = z.object({
  stats: z.record(z.string(), z.number()),   // dict keyed by canonical stat name
  format_detected: z.enum(["aggregated", "individual"]),
  row_count: z.number(),
  warnings: z.array(z.string()),
})
```

**New API client functions** in `apps/frontend/dashboard/src/lib/api-client.ts`:

Add a `apiGet` helper (GET requests currently missing — `apiFetch` hardcodes POST):

```typescript
async function apiGet<T>(path: string, schema: z.ZodType<T>): Promise<T>
```

Then add:
- `listHerdProfiles()` — uses `apiGet`
- `getHerdProfile(id: number)` — uses `apiGet`
- `createHerdProfile(data: HerdProfileCreate)` — uses `apiFetch` (POST)
- `updateHerdProfile(id: number, data: HerdProfileCreate)` — needs a `apiPut` helper
- `deleteHerdProfile(id: number)` — needs `fetch` with `method: "DELETE"`, returns `void`
- `uploadHerdProfileCsv(file: File)` — uses `FormData`, not JSON; needs its own fetch call

**Navigation entry** in `apps/frontend/dashboard/src/components/dashboard/navigation.ts`:
```typescript
{ label: "Herd Stats", href: "/herd-stats", icon: BarChart3 }
```

---

### 6. Autoencoder page integration

The `HerdStatsForm` sliders are rendered in:
- **Component:** `apps/frontend/dashboard/src/app/(dashboard)/curves/components/autoencoder-input-panel.tsx` (line 145)
- **State:** `herdStats` / `setHerdStats` managed in `apps/frontend/dashboard/src/app/(dashboard)/curves/page.tsx` (line 136)

The `autoencoder/page.tsx` is a redirect stub — do not modify it.

Integration: add a `Select` dropdown in `autoencoder-input-panel.tsx` above the `HerdStatsForm`. The dropdown lists saved herd profiles (fetched via `useHerdProfiles`). Selecting a profile calls `onHerdStatsChange` with the profile's 10 values mapped via `HERD_STATS_METADATA` order. A "None (manual)" option restores previous behavior.

The `herdStats` state in `curves/page.tsx` may need to be passed down to `autoencoder-input-panel.tsx` via props, or `useHerdProfiles` can be called directly in the panel — whichever is cleaner given the existing prop structure.

---

## Data flow: CSV upload to prediction

```
User uploads CSV
    ↓
POST /herd-profiles/csv-preview  (Central API)
    ↓ parse_csv() → raw dict[str, float]
    ↓ HerdStatsRangeNormalizationTransform → dict[str, float] (0-1)
    ← HerdProfileUploadResponse (preview, not saved)
        ↓
User reviews preview, adjusts sliders if needed
        ↓
POST /herd-profiles/  (save to DB with name)
        ↓
User selects profile on curves page (dropdown in autoencoder-input-panel)
        ↓
POST /autoencoder/predict  (herd_stats: [10 normalized floats])
        ↓
TF model inference
```

---

## Phased implementation

### Phase 1 — CRUD + manual entry (independently shippable)
1. Add `HerdProfile` to `models.py` + Alembic migration
2. `routes/herd_profiles.py` with CRUD (no upload yet)
3. Register router in `app.py`
4. Add `apiGet`/`apiPut`/`apiDelete` helpers to `api-client.ts`
5. Frontend: herd-stats page with list + create/edit form
6. Navigation entry
7. Autoencoder panel: profile dropdown

### Phase 2 — CSV upload
1. `HerdStatsRangeNormalizationTransform` in `lactation_transforms.py`
2. `herd_stats_ranges` section in config.yaml
3. `herd_stats_ingestion.py` utility (parse_csv + normalize)
4. `csv-preview` endpoint in `routes/herd_profiles.py`
5. Frontend: upload component with preview

### Phase 3 — Polish
1. Extended column alias mapping (common CSV header variations)
2. Calibrate normalization ranges against actual training data statistics
3. Edge case hardening: out-of-range clamping warnings, partial column fills

---

## Test plan

**Backend (Phase 1):** `cd apps/backend/api && just test`
- CRUD happy path: create, read, update, delete
- Duplicate name → 409
- Invalid stat values (>1.0) → 422

**Backend (Phase 2):** add to same suite
- `parse_csv`: aggregated format (1 row), individual format (mean across rows)
- `parse_csv`: partial columns (some missing) → returns warnings, not error
- `parse_csv`: zero matching columns → raises `ValueError`
- `parse_csv`: unparseable bytes → raises `ValueError`
- `parse_csv`: out-of-range raw values → normalize clamps to [0,1]
- Upload endpoint: file > 10 MB → 413

**Model package:** `cd packages/models/lactation-autoencoder && just test`
- `HerdStatsRangeNormalizationTransform`: verify normalization with known ranges
- Verify canonical order of output matches `HERD_STATS_METADATA` in frontend

**End-to-end manual test:**
1. `just run-api` + `just run-dashboard`
2. Create herd profile via management page
3. Select it on curves page → verify sliders update
4. Run prediction → verify herd stats propagate correctly
5. Upload aggregated CSV → check preview
6. Upload individual cow CSV → check mean aggregation + preview

---

## Critical files

| File | Change |
|------|--------|
| `packages/models/lactation-autoencoder/src/lactation_autoencoder/dataloaders/transforms/lactation_transforms.py` | Add `HerdStatsRangeNormalizationTransform` (not registered) |
| `data/experiments/lactation_autoencoder/versions/v15/config/config.yaml` | Add `herd_stats_ranges` section |
| `apps/backend/api/src/bovi_api/herd_stats_ingestion.py` | New file: CSV parsing + normalization utility |
| `apps/backend/api/src/bovi_api/models.py` | Add `HerdProfile` SQLModel |
| `apps/backend/api/src/bovi_api/routes/herd_profiles.py` | New CRUD + csv-preview router |
| `apps/backend/api/src/bovi_api/app.py` | Register herd_profiles router |
| `apps/backend/api/alembic/versions/` | New migration for herd_profiles table |
| `apps/frontend/dashboard/src/types/api.ts` | Zod schemas for HerdProfile |
| `apps/frontend/dashboard/src/lib/api-client.ts` | Add GET/PUT/DELETE helpers + herd profile functions |
| `apps/frontend/dashboard/src/app/(dashboard)/herd-stats/` | New page + components + hooks |
| `apps/frontend/dashboard/src/app/(dashboard)/curves/components/autoencoder-input-panel.tsx` | Profile dropdown integration |
| `apps/frontend/dashboard/src/components/dashboard/navigation.ts` | Navigation entry |
