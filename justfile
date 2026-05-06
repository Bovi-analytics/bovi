set dotenv-load

# ── Derived URLs (from PORT_* in .env) ───────────────────────
export LACTATION_CURVES_URL := "http://localhost:" + env("PORT_CURVES", "8001")
export LACTATION_AUTOENCODER_URL := "http://localhost:" + env("PORT_AUTOENCODER", "8002")
export NEXT_PUBLIC_API_URL := "http://localhost:" + env("PORT_API", "8000")
export CORS_ORIGINS := "[\"http://localhost:" + env("PORT_DASHBOARD", "3000") + "\"]"

# ── Workspace ────────────────────────────────────────────────
sync:
    uv sync --all-packages
    cd apps/infrastructure && uv sync
    git config core.hooksPath .githooks

lint:
    uv run ruff check --fix && uv run ruff format

test:
    uv run pytest -v

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

# Apply DB migrations manually (the API also does this automatically on startup)
db-migrate:
    cd apps/backend/api && uv run alembic upgrade head

# ── Containerised API (SQLite on a Docker volume) ────────────
api-build:
    cd apps/backend/api && docker compose build

api-up:
    cd apps/backend/api && docker compose up -d

api-down:
    cd apps/backend/api && docker compose down

api-logs:
    cd apps/backend/api && docker compose logs -f

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
preview-infra:
    cd apps/infrastructure && pulumi preview

deploy-infra:
    cd apps/infrastructure && pulumi up
