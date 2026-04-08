# Bovi

Monorepo for the Bovi dairy analytics platform.

## Structure

### Packages (libraries)

- `packages/bovi-core/` — ML framework: base classes, registries, config, utilities
- `packages/models/lactationcurve/` — Classical lactation curve fitting (Wood, MilkBot, etc.)
- `packages/models/lactation-autoencoder/` — TensorFlow autoencoder for milk production prediction
- `packages/models/bovi-yolo/` — YOLO object detection for dairy applications
- `packages/infrastructure/pulumi/` — Azure infrastructure as code

### Apps (deployables)

- `apps/backend/api/` — Central FastAPI gateway with PostgreSQL persistence
- `apps/backend/models/lactation-curves/` — Azure Function App for classical curve fitting
- `apps/backend/models/lactation-autoencoder/` — Azure Function App for TF autoencoder
- `apps/frontend/dashboard/` — Next.js visualization dashboard

## Getting Started

```bash
just sync        # Install dependencies + set up pre-commit hooks
just test        # Run all tests
just dev         # Start all services (API, models, dashboard)
```

`just sync` automatically configures git to use the checked-in pre-commit hooks (`.githooks/`), which run linting, formatting, type checking, and secrets detection on every commit.

> **Optional:** If you use [direnv](https://direnv.net/), hooks are set up automatically when you `cd` into the repo.

## Requirements

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **[just](https://just.systems/)** — task runner ([install options](https://just.systems/man/en/packages.html))
  - macOS: `brew install just`
  - Windows: `winget install Casey.Just`, or download from [releases](https://github.com/casey/just/releases)
- **[bun](https://bun.sh/)** — dashboard only
  - macOS/Linux: `curl -fsSL https://bun.sh/install | bash`
  - Windows: `powershell -c "irm bun.sh/install.ps1 | iex"`
