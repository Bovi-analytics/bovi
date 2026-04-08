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
uv sync          # Install all Python dependencies
just test        # Run all tests
just run-api     # Start the central API
just run-dashboard  # Start the dashboard (requires bun) 
```

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [bun](https://bun.sh/) (for the dashboard)
- [just](https://just.systems/) (task runner)
