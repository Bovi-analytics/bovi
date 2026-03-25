# ── Development ──────────────────────────────────────────────
sync:
    uv sync --all-packages

lint:
    uv run ruff check --fix && uv run ruff format

test:
    uv run pytest -v

# ── Packages ────────────────────────────────────────────────
test-core:
    uv run pytest packages/bovi-core/tests/ -v

test-lactationcurve:
    uv run pytest packages/models/lactationcurve/tests/ -v

test-autoencoder:
    uv run pytest packages/models/lactation-autoencoder/tests/ -v

test-yolo:
    uv run pytest packages/models/bovi-yolo/tests/ -v

build-lactationcurve:
    cd packages/models/lactationcurve && uv build

publish-lactationcurve:
    #!/usr/bin/env bash
    set -euo pipefail
    set -a && source .env && set +a
    cd packages/models/lactationcurve && rm -rf dist && uv build && uv publish

# ── Backend Apps ─────────────────────────────────────────────
run-api:
    cd apps/backend/api && uv run python -m uvicorn bovi_api.app:app --reload

test-api:
    uv run pytest apps/backend/api/tests/ -v

test-lactation-curves-app:
    uv run pytest apps/backend/models/lactation-curves/tests/ -v

test-autoencoder-app:
    uv run pytest apps/backend/models/lactation-autoencoder/tests/ -v

# ── Frontend ─────────────────────────────────────────────────
run-dashboard:
    cd apps/frontend/dashboard && bun dev

# ── Infrastructure ──────────────────────────────────────────
preview-infra:
    cd packages/infrastructure/pulumi && pulumi preview

deploy-infra:
    cd packages/infrastructure/pulumi && pulumi up
