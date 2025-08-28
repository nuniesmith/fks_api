# FKS API Service

Lightweight FastAPI service providing HTTP/WebSocket endpoints for the FKS platform.

## Features

- Health & info endpoints
- Synthetic chart + indicator data for UI development
- Optional background Ollama model pull (dev)
- Modular router loading with graceful degradation

## Quick Start

```bash
# Poetry (recommended)
poetry install --no-root
poetry run uvicorn fks_api.fastapi_main:app --reload --port 8000

# Or plain venv + pip
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install .[websocket,security]
uvicorn fks_api.fastapi_main:app --reload --port 8000
```

Visit: <http://localhost:8000/docs>

## Smoke Test

```bash
poetry run pytest -q
```

## Environment Vars

- API_SERVICE_NAME, API_SERVICE_PORT
- APP_ENV (development|production)
- OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_FAST_MODEL

## Docker

```bash
# Dev (build + live reload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Production (pre-built images)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Pull-only sanity
docker compose -f docker-compose.yml -f docker-compose.pull-only.yml pull
```

Celery example task:

```bash
docker compose exec worker celery -A app.celery call fks_api.ping
```

## Next Steps

- Replace synthetic data endpoints with real data service
- AuthN/Z middleware integration
- Metrics & tracing
- Deeper Celery task modules / scheduling

