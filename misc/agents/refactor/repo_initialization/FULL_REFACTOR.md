# Plan: Create the `bovi` Monorepo (Revised)

## Context

The bovi ecosystem is currently spread across multiple independent repos (bovi-core, bovi-models-douwe, lactation_curve_core). The goal is to consolidate the public framework, public models, apps, and infrastructure into a single `bovi` monorepo — while keeping private models in a separate `bovi-private` repo.

This enables:
- The lactation autoencoder (Arno's model) to be served via API using the bovi-core framework
- Classical curve fitting (lactationcurve) alongside the autoencoder
- YOLO object detection models alongside lactation models
- A unified dashboard showing all model outputs
- A central API with persistence (PostgreSQL) and monitoring
- Clean public/private separation

## Architecture Decisions

| Decision | Choice |
|----------|--------|
| Repo name | `bovi` (public) |
| Python version | 3.12 |
| Private models | Separate `bovi-private` repo |
| bovi-core distribution | Published to PyPI (slim — no ML framework deps) |
| bovi-core deps | pyyaml, dotenv, numpy, requests, azure-storage-blob only |
| MilkBot app | Merged into lactation-curves app |
| Model weights/pickles | Azure Blob Storage (downloaded at startup) |
| bovi-models-template | Stays separate (minimal skeleton, tutorials live here) |
| bovi-models-douwe | Renamed to `bovi-models-example` (separate repo) |
| YOLO model | Included in monorepo as packages/models/bovi-yolo/ |
| Apps layout | Split into apps/frontend/ and apps/backend/ |
| API architecture | Central API (gateway/persistence) + separate model Function Apps |
| Database | PostgreSQL (Azure) |
| Dashboard routing | Dashboard → Central API only (never calls model apps directly) |
| Dashboard pkg manager | bun |
| uv workspace | Backend Python packages only (dashboard excluded) |

## Repo Structure

```
bovi/
├── .github/
│   └── workflows/
├── packages/
│   ├── bovi-core/
│   │   ├── src/bovi_core/
│   │   │   ├── config/
│   │   │   ├── types/
│   │   │   ├── ml/
│   │   │   │   ├── dataloaders/
│   │   │   │   ├── models/
│   │   │   │   └── predictors/
│   │   │   └── utils/
│   │   ├── tests/
│   │   └── pyproject.toml
│   ├── models/
│   │   ├── lactation-autoencoder/
│   │   │   ├── src/lactation_autoencoder/
│   │   │   │   ├── dataloaders/
│   │   │   │   │   ├── sources/
│   │   │   │   │   ├── datasets/
│   │   │   │   │   └── transforms/
│   │   │   │   ├── models/
│   │   │   │   └── predictors/
│   │   │   ├── tests/
│   │   │   └── pyproject.toml
│   │   ├── lactationcurve/
│   │   │   ├── src/lactationcurve/
│   │   │   │   ├── fitting/
│   │   │   │   ├── characteristics/
│   │   │   │   └── preprocessing/
│   │   │   ├── tests/
│   │   │   └── pyproject.toml
│   │   └── bovi-yolo/
│   │       ├── src/bovi_yolo/
│   │       │   ├── dataloaders/
│   │       │   │   ├── sources/
│   │       │   │   ├── datasets/
│   │       │   │   └── transforms/
│   │       │   ├── models/
│   │       │   └── predictors/
│   │       ├── tests/
│   │       └── pyproject.toml
│   └── infrastructure/
│       └── pulumi/
│           ├── src/
│           │   └── __main__.py
│           └── pyproject.toml
├── apps/
│   ├── frontend/
│   │   └── dashboard/
│   │       ├── src/
│   │       ├── package.json
│   │       └── .env.local
│   └── backend/
│       ├── api/
│       │   ├── main.py
│       │   ├── tests/
│       │   └── pyproject.toml
│       └── models/
│           ├── lactation-curves/
│           │   ├── function_app.py
│           │   ├── main.py
│           │   ├── host.json
│           │   ├── tests/
│           │   └── pyproject.toml
│           └── lactation-autoencoder/
│               ├── function_app.py
│               ├── main.py
│               ├── host.json
│               ├── tests/
│               └── pyproject.toml
├── misc/
│   ├── agents/
│   └── users/
├── pyproject.toml
├── justfile
├── CLAUDE.md
├── .gitignore
└── README.md
```

## Data Flow

```
Dashboard (Next.js) → Central API (apps/backend/api/) → PostgreSQL (Azure)
                          │
                          ├→ lactation-curves/       (internal Azure Function)
                          └→ lactation-autoencoder/  (internal Azure Function)
```

## External Repos (unchanged, separate)

```
bovi-models-template/     # Minimal skeleton for new users (tutorials live here)
bovi-models-example/      # Template with worked examples (renamed from bovi-models-douwe)
bovi-private/             # Private models (separate private repo, same structure)
```

These depend on `bovi-core` via PyPI.

## Implementation Phases

### Phase 1: Repo Scaffolding ✅
1. Create `bovi/` directory with full folder structure
2. Create root `pyproject.toml` (uv workspace — backend only)
3. Create `justfile` with commands for all packages and apps (bun for dashboard)
4. Create `CLAUDE.md` with project conventions
5. Create `.gitignore` and `README.md`
6. Create `.github/workflows/` for CI/CD

### Phase 2: Move bovi-core
1. Copy `src/bovi_core/` from current bovi-core repo into `packages/bovi-core/src/`
2. Copy tests into `packages/bovi-core/tests/`
3. Slim pyproject.toml — Python 3.12, only framework deps (no torch/tf/databricks)
4. Verify: `uv run pytest packages/bovi-core/tests/`

### Phase 3: Move lactationcurve
1. Copy `packages/python/lactation/src/lactationcurve/` from lactation_curve_core
2. Copy tests
3. Copy `pyproject.toml`, adapt paths
4. Verify: `uv run pytest packages/models/lactationcurve/tests/`

### Phase 4: Move lactation-autoencoder
1. Copy model code from `bovi-models-douwe/src/models/lactation/`
2. Adapt imports: `from models.lactation.xxx` → `from lactation_autoencoder.xxx`
3. Copy tests, adapt imports
4. Verify: `uv run pytest packages/models/lactation-autoencoder/tests/`

### Phase 5: Move YOLO
1. Copy model code from `bovi-models-douwe/src/models/yolo/`
2. Adapt imports: `from models.yolo.xxx` → `from bovi_yolo.xxx`
3. Copy tests, adapt imports
4. Verify: `uv run pytest packages/models/bovi-yolo/tests/`

### Phase 6: Create Backend Apps
1. Build `apps/backend/api/` — Central FastAPI gateway with PostgreSQL persistence
2. Build `apps/backend/models/lactation-curves/` — Azure Function App (classical fitting + milkbot)
3. Build `apps/backend/models/lactation-autoencoder/` — Azure Function App (TF model)
4. Verify: `uv run pytest apps/backend/*/tests/`

### Phase 7: Move dashboard
1. Copy `apps/dashboard/` from lactation_curve_core into `apps/frontend/dashboard/`
2. Update API client to point to central API only (env var: NEXT_PUBLIC_API_URL)
3. Verify: `bun dev` starts, pages load

### Phase 8: Move infrastructure
1. Copy Pulumi code from `lactation_curve_core/packages/python/infrastructure/`
2. Adapt for new app structure (separate Function App resources per model app)
3. Add PostgreSQL resource for central API
4. Verify: `pulumi preview`

## Key Source Files (to copy from)

| Source | Destination |
|--------|-------------|
| `bovi-core/src/bovi_core/` | `packages/bovi-core/src/bovi_core/` |
| `bovi-core/tests/` | `packages/bovi-core/tests/` |
| `lactation_curve_core/packages/python/lactation/src/lactationcurve/` | `packages/models/lactationcurve/src/lactationcurve/` |
| `lactation_curve_core/packages/python/lactation/tests/` | `packages/models/lactationcurve/tests/` |
| `bovi-models-douwe/src/models/lactation/` | `packages/models/lactation-autoencoder/src/lactation_autoencoder/` |
| `bovi-models-douwe/tests/models/lactation/` | `packages/models/lactation-autoencoder/tests/` |
| `bovi-models-douwe/src/models/yolo/` | `packages/models/bovi-yolo/src/bovi_yolo/` |
| `bovi-models-douwe/tests/models/yolo/` | `packages/models/bovi-yolo/tests/` |
| `lactation_curve_core/apps/lactation_curves/main.py` | `apps/backend/models/lactation-curves/main.py` (adapted) |
| `lactation_curve_core/apps/milkbot/main.py` | Merged into `apps/backend/models/lactation-curves/main.py` |
| `lactation_curve_core/apps/dashboard/` | `apps/frontend/dashboard/` |
| `lactation_curve_core/packages/python/infrastructure/` | `packages/infrastructure/pulumi/src/` |

## Model Weights & Data

- Weights and pickle files stored in **Azure Blob Storage**
- Model apps download them at startup (cached locally)
- Env vars per app: `MODEL_BLOB_URL`, `HERD_STATS_BLOB_URL`
- For local dev: `MODEL_PATH` / `HERD_STATS_PATH` pointing to local directories

## Verification

1. `uv sync` — workspace resolves all dependencies
2. `uv run pytest packages/bovi-core/tests/` — framework tests pass
3. `uv run pytest packages/models/lactationcurve/tests/` — classical fitting tests pass
4. `uv run pytest packages/models/lactation-autoencoder/tests/` — autoencoder tests pass
5. `uv run pytest packages/models/bovi-yolo/tests/` — YOLO tests pass
6. `uv run pytest apps/backend/api/tests/` — central API tests pass
7. `uv run pytest apps/backend/models/lactation-curves/tests/` — curves app tests pass
8. `uv run pytest apps/backend/models/lactation-autoencoder/tests/` — autoencoder app tests pass
9. `just run-api` — central API starts, `GET /` returns ok
10. `just run-dashboard` — dashboard starts (bun), pages load
11. `uv run ruff check --fix && uv run ruff format` — all code passes linting
