# AGENTS.md

## Project

bovi is a monorepo for the Bovi dairy analytics platform.

Key areas:

- `packages/bovi-core/`: slim ML framework and shared utilities.
- `packages/models/lactation-autoencoder/`: TensorFlow lactation prediction model.
- `packages/models/lactationcurve/`: classical lactation curve fitting models.
- `packages/models/bovi-yolo/`: YOLO object detection for dairy applications.
- `apps/backend/api/`: central FastAPI gateway.
- `apps/backend/models/`: deployable backend model apps.
- `apps/frontend/dashboard/`: Next.js dashboard.

## Required Workflow

Use the user-level `$worktree-first` skill for new implementation work. Create worktrees under `.worktrees/` so parallel work does not interfere with the main checkout.

Use the user-level `$write-tests` skill whenever adding functionality, fixing bugs, refactoring behavior, or touching code with existing tests.

Commit cadence:

- For many tasks or broad changes, make intermittent commits at coherent, tested checkpoints.
- For small focused tasks, one commit is fine.
- Do not commit unrelated user changes.

## Tests

The final regression command for this repo is:

```bash
just test
```

Run targeted tests while developing when useful, but run `just test` before committing unless the environment prevents it. If it cannot be run, report the exact reason and the closest test command that was run.

When adding functionality:

- Reuse existing `conftest.py` files and fixtures first.
- Add new fixtures only when existing fixtures do not fit.
- Update existing tests first when touched code changes existing behavior.
- Add new tests after checking and updating old tests.
- Add integration tests when behavior crosses package, API, database, service, or app boundaries.

## Commands

From repo root:

```bash
just sync
just test
just lint
just run-api
just run-dashboard
```

Per-package examples:

```bash
cd packages/bovi-core && just test
cd packages/models/bovi-yolo && just test
cd packages/models/lactationcurve && just test
cd apps/backend/api && just test
cd apps/backend/models/lactation-curves && just test
```

## Constraints

- Python 3.12 only.
- Use `uv` for Python dependency management.S
- Use `bun` for the dashboard.
- Import from `bovi_core`, never `src.bovi_core`.
- Register models with `@ModelRegistry.register("name")`.
- Do not commit model weights.
- Keep `bovi-core` slim; do not add ML framework dependencies there.
- Dashboard must talk to the central API only, never directly to model apps.
