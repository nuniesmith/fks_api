## ============================================================================
## Dockerfile - FKS API (multi-stage, poetry based)
## ----------------------------------------------------------------------------
## Stages:
##  1. base    : shared system deps + poetry
##  2. builder : install deps (no dev) + project wheel build
##  3. runtime : slim image with only runtime deps + app code
## ----------------------------------------------------------------------------
## Usage:
##  docker build -t fks-api:dev .
##  docker run -p 8000:8000 fks-api:dev
## ============================================================================

ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION} AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /app

# Copy only dependency metadata first (improves layer caching)
COPY src/pyproject.toml pyproject.toml
COPY README.md README.md

FROM base AS builder

# Export/lock dependencies (will generate poetry.lock if absent)
RUN if [ ! -f poetry.lock ]; then poetry lock --no-update || true; fi

RUN poetry install --no-root --no-dev

# Copy source code
COPY sitecustomize.py sitecustomize.py
COPY src ./src
COPY shared ./shared

# Install project (build wheel inside container for proper entrypoints if any)
RUN poetry install --no-dev

FROM python:${PYTHON_VERSION} AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Copy sitecustomize first (path bootstrap)
COPY sitecustomize.py sitecustomize.py

# Copy installed environment from builder (site-packages)
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy source (keep editable feel for runtime config/templates)
COPY src ./src
COPY shared ./shared

EXPOSE 8000

# Default command (can be overridden by compose for worker/scheduler)
CMD ["uvicorn", "fks_api.fastapi_main:app", "--host", "0.0.0.0", "--port", "8000"]
