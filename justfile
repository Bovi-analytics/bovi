set dotenv-load

# ── Derived URLs (from PORT_* in .env) ───────────────────────
export PORT_API := env("PORT_API", "8000")
export PORT_CURVES := env("PORT_CURVES", "8001")
export PORT_AUTOENCODER := env("PORT_AUTOENCODER", "8002")
export PORT_DASHBOARD := env("PORT_DASHBOARD", "3000")

export LACTATION_CURVES_URL := "http://localhost:" + PORT_CURVES
export LACTATION_AUTOENCODER_URL := "http://localhost:" + PORT_AUTOENCODER
export NEXT_PUBLIC_API_URL := "http://localhost:" + PORT_API
export CORS_ORIGINS := "[\"http://localhost:" + PORT_DASHBOARD + "\"]"

# ── Workspace ────────────────────────────────────────────────
sync:
    uv sync --all-packages
    git config core.hooksPath .githooks

lint:
    uv run ruff check --fix && uv run ruff format

test:
    uv run python scripts/test_affected.py

test-affected base="origin/main":
    uv run python scripts/test_affected.py --base {{base}}

test-affected-dry-run base="origin/main":
    uv run python scripts/test_affected.py --base {{base}} --dry-run

test-fast:
    uv run python scripts/test_affected.py --fast

test-all:
    uv run pytest -c pyproject.toml -v

# ── Run services ─────────────────────────────────────────────
check-ports:
    #!/usr/bin/env bash
    set -euo pipefail
    failed=0
    for pair in \
        "$PORT_API:Central API" \
        "$PORT_CURVES:Lactation Curves" \
        "$PORT_AUTOENCODER:Lactation Autoencoder" \
        "$PORT_DASHBOARD:Dashboard"; do
        port="${pair%%:*}"
        name="${pair#*:}"
        pid=$(lsof -ti :"$port" 2>/dev/null || true)
        if [ -n "$pid" ]; then
            echo "ERROR: Port $port ($name) is occupied by PID $pid"
            failed=1
        fi
    done
    if [ "$failed" -eq 1 ]; then
        echo ""
        echo "Run 'just stop' to free occupied ports, or 'just dev' to stop-then-start."
        exit 1
    fi
    echo "All ports available."

run: check-ports
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT
    just run-api 2>&1 | sed $'s/^/\033[34m[API]\033[0m /' &
    just run-models 2>&1 &
    just run-dashboard 2>&1 | sed $'s/^/\033[35m[WEB]\033[0m /' &
    wait

stop:
    #!/usr/bin/env bash
    echo "Stopping services..."
    for pair in \
        "$PORT_API:API" \
        "$PORT_CURVES:Curves" \
        "$PORT_AUTOENCODER:Autoencoder" \
        "$PORT_DASHBOARD:Dashboard"; do
        port="${pair%%:*}"
        name="${pair#*:}"
        lsof -ti :"$port" | xargs kill -9 2>/dev/null \
            && echo "Killed $name (port $port)" \
            || echo "$name not running (port $port)"
    done

dev: stop run

run-api:
    cd apps/backend/api && uv run python -m uvicorn bovi_api.app:app --reload --port $PORT_API

# Apply DB migrations manually before rolling a new API revision.
db-migrate:
    cd apps/backend/api && uv run python -m bovi_api.migrations

# ── Containerised API (SQLite on a Docker volume) ────────────
api-build:
    cd apps/backend/api && docker compose build

api-up:
    cd apps/backend/api && docker compose up -d

api-down:
    cd apps/backend/api && docker compose down

api-logs:
    cd apps/backend/api && docker compose logs -f

# Build, start, hit /health, then stop — quick smoke test
api-smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    cd apps/backend/api
    docker compose build
    docker compose up -d
    echo "Waiting for API to be ready..."
    for i in $(seq 1 15); do
        if curl -sf http://localhost:8000/ > /dev/null 2>&1; then
            echo "✓ / OK"
            docker compose down
            exit 0
        fi
        sleep 1
    done
    echo "✗ /health did not respond in time"
    docker compose down
    exit 1

# ── Containerised stack (migration job + API + dashboard) ───
compose-build:
    docker compose build

compose-up:
    docker compose up -d

compose-down:
    docker compose down

compose-logs:
    docker compose logs -f

# Build, start, hit API + dashboard, then stop — quick smoke test
compose-smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    docker compose build
    docker compose up -d
    trap 'docker compose down >/dev/null 2>&1 || true' EXIT
    echo "Waiting for API and dashboard to be ready..."
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:$PORT_API/" >/dev/null 2>&1; then
            echo "✓ API / OK"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "✗ API / did not respond in time"
            exit 1
        fi
        sleep 1
    done
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:$PORT_DASHBOARD/" >/dev/null 2>&1; then
            echo "✓ Dashboard / OK"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "✗ Dashboard / did not respond in time"
            exit 1
        fi
        sleep 1
    done
    if curl -sf "http://localhost:$PORT_DASHBOARD/api/bovi/health" >/dev/null 2>&1; then
        echo "✓ Dashboard API proxy /api/bovi/health OK"
        exit 0
    fi
    echo "✗ Dashboard API proxy did not respond"
    exit 1

run-models:
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT
    cd apps/backend/models/lactation-curves && just run 2>&1 | sed $'s/^/\033[32m[CURVES]\033[0m /' &
    cd apps/backend/models/lactation-autoencoder && just run 2>&1 | sed $'s/^/\033[33m[AUTOENC]\033[0m /' &
    wait

run-dashboard:
    #!/usr/bin/env bash
    set -euo pipefail
    cd apps/frontend/dashboard
    [ ! -d node_modules ] && bun install
    bun dev --port $PORT_DASHBOARD

# ── Infrastructure ──────────────────────────────────────────
infra-preview:
    #!/usr/bin/env bash
    set -euo pipefail
    cd apps/infrastructure
    source scripts/load-env.sh
    pulumi preview

infra-deploy:
    #!/usr/bin/env bash
    set -euo pipefail
    cd apps/infrastructure
    source scripts/load-env.sh
    pulumi up --yes
