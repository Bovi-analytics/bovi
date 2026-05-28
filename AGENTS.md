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

Before inspecting or editing for implementation work, use the user-level
`$tmux-shell` skill. Choose a concise task name, rename the tmux session and
window with that name, enable terminal-title propagation, and rename the Codex
thread with the same task name using the skill's SQLite command. Do not rely on
`/rename`; Codex cannot execute TUI slash commands itself. Verify the tmux
display message and the Codex thread title query before continuing.

After entering the Bovi worktree, update the tmux `@codex_context` header with
the branch, worktree name, and base commit. Verify the value before running repo
commands so the header reflects Codex's actual worktree, not just the shell's
original directory.

Run `just sync` from the worktree immediately after entering it and before any
tests, linting, typechecking, commit hooks, or import-dependent debugging. This
is mandatory for every new or reused worktree. If a command fails because the
worktree was not synced, treat that as setup error: run `just sync`, then rerun
the failed command.

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

`just test` is the required final regression command and currently delegates to
the affected-test runner. Use `just test-all` only when an explicit full pytest
run is needed. Do not replace the required `just test` with package-local tests
unless the environment prevents the repo command from running.

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
