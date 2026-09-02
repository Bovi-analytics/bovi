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

# Git metadata is intentionally excluded from the Docker build context. Supply a
# PEP 440 version so hatch-vcs can still build the lactationcurve dependency.
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+docker

COPY packages/bovi-core/pyproject.toml ./packages/bovi-core/
COPY packages/bovi-core/src ./packages/bovi-core/src
COPY packages/models/lactationcurve/pyproject.toml ./packages/models/lactationcurve/
COPY packages/models/lactationcurve/README.md ./packages/models/lactationcurve/
COPY packages/models/lactationcurve/src ./packages/models/lactationcurve/src
COPY apps/backend/api/pyproject.toml ./apps/backend/api/
COPY apps/backend/api/src ./apps/backend/api/src
COPY apps/backend/api/scripts ./apps/backend/api/scripts

RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION}" \
    uv pip install --system \
        ./packages/bovi-core \
        ./packages/models/lactationcurve \
        ./apps/backend/api

# SQLite file lives on a mounted volume.
ENV DATABASE_URL=sqlite+aiosqlite:////data/bovi.db
VOLUME ["/data"]

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "bovi_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
