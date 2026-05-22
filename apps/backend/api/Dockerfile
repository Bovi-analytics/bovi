# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip compile pyproject.toml --no-header --no-annotate -o /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt

COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps .

# SQLite file lives on a mounted volume.
ENV DATABASE_URL=sqlite+aiosqlite:////data/bovi.db
VOLUME ["/data"]

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "bovi_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
