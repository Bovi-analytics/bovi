# CLAUDE.md

## Project

bovi is a monorepo for the Bovi dairy analytics platform containing:

### Packages (libraries)
- **packages/bovi-core/** — Slim ML framework: base classes, registries, config, utilities (published to PyPI)
- **packages/models/lactation-autoencoder/** — TensorFlow autoencoder for milk production prediction
- **packages/models/lactationcurve/** — Classical lactation curve fitting: Wood, MilkBot, Wilmink, etc. (published to PyPI)
- **packages/models/bovi-yolo/** — YOLO object detection for dairy applications
- **packages/infrastructure/pulumi/** — Azure infrastructure as code

### Apps (deployables)
- **apps/backend/api/** — Central FastAPI gateway: unified contract, persistence (PostgreSQL), monitoring
- **apps/backend/models/lactation-curves/** — Azure Function App: classical curve fitting + milkbot
- **apps/backend/models/lactation-autoencoder/** — Azure Function App: TF autoencoder predictions
- **apps/frontend/dashboard/** — Next.js visualization dashboard (bun)

### Data flow
Dashboard → Central API → PostgreSQL + proxies to model Function Apps (internal)

## Commands

```bash
uv sync --all-packages           # Install all workspace dependencies
uv run pytest                    # Run all tests
uv run ruff check --fix && uv run ruff format  # Lint and format
just run-api                     # Run central API locally
just run-dashboard               # Run dashboard locally (bun)
just test-core                   # Test bovi-core only
just test-lactationcurve         # Test lactationcurve only
just test-autoencoder            # Test autoencoder library only
just test-yolo                   # Test bovi-yolo only
just test-api                    # Test central API only
just test-lactation-curves-app   # Test lactation-curves Function App
just test-autoencoder-app        # Test autoencoder Function App
```

## Architecture

```
External repos (depend on bovi-core via PyPI):
  bovi-models-template    — Minimal skeleton for new users
  bovi-models-example     — Template with worked examples
  bovi-private            — Private models (separate private repo)
```

## Critical Rules

- Python 3.12 only
- Import from `bovi_core`, never `src.bovi_core` (breaks singletons)
- Register models with `@ModelRegistry.register("name")`
- Model weights stored in Azure Blob Storage, never committed to git
- Follow PEP8, use ruff for formatting, basedpyright for type checking
- Use `uv` as Python package manager, `bun` for frontend
- bovi-core must stay slim — no ML framework deps (torch, tf, etc.)
- Each backend model app is independently deployable
- Dashboard talks to central API only, never directly to model apps
