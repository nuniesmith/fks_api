# FKS API Service

Lightweight FastAPI service providing HTTP/WebSocket endpoints for the FKS platform.

## Features

- Health & info endpoints
- Synthetic chart + indicator data for UI development
- Optional background Ollama model pull (dev)
- Modular router loading with graceful degradation

## Quick Start (Standalone)

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

## Modes

This service can run in two modes:

1. **Standalone (default)** – Uses lightweight stubs (`standalone_shared.py`) and does not require the external shared monorepo packages.
2. **Shared Mode** – Mounts shared repositories and sets `USE_SHARED=1` to prefer real shared utilities.

Switch via environment variable:

```bash
# Standalone (default)
docker compose up

# Shared
docker compose -f docker-compose.yml -f docker-compose.shared.yml up
```

Or manually:

```bash
USE_SHARED=1 uvicorn fastapi_main:app --reload
```

## Docker

```bash
# Dev (standalone editable install)
docker compose up --build

# Production (non-editable, lean image)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Pull-only sanity
docker compose -f docker-compose.yml -f docker-compose.pull-only.yml pull

# Build w/o editable install (alternative)
docker compose build --build-arg INSTALL_EDITABLE=0 fks_api
```

### Shared Mode Example

```bash
docker compose -f docker-compose.yml -f docker-compose.shared.yml up --build
```

This mounts `shared/` subdirectories and sets `USE_SHARED=1` so the runtime imports the real shared utilities instead of stubs.

Celery example task:

```bash
docker compose exec worker celery -A app.celery call fks_api.ping
```

## 🔧 Configuration

### Environment Variables

```bash
# Service Configuration
API_SERVICE_NAME=fks_api
API_SERVICE_PORT=8001
APP_ENV=production  # development|production

# Ollama Configuration (Optional)
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3
OLLAMA_FAST_MODEL=llama3:8b

# Shared Mode
USE_SHARED=0  # 0=standalone, 1=shared mode
```

## 🧪 Testing

```bash
# Run tests
poetry run pytest -q

# Or with pip
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 🐳 Docker

### Build

```bash
docker build -t nuniesmith/fks:api-latest .
```

### Run

```bash
docker run -p 8001:8001 \
  -e APP_ENV=production \
  nuniesmith/fks:api-latest
```

## ☸️ Kubernetes

### Deployment

```bash
# Deploy using Helm
cd repo/main/k8s/charts/fks-platform
helm install fks-platform . -n fks-trading

# Or using the unified start script
cd /home/jordan/Documents/code/fks
./start.sh --type k8s
```

### Health Checks

Kubernetes probes:
- **Liveness**: `GET /live`
- **Readiness**: `GET /ready`

## 📚 Documentation

- [API Documentation](docs/API.md) - Complete API reference
- [Deployment Guide](docs/DEPLOYMENT.md) - Deployment instructions
- [Architecture Details](docs/ARCHITECTURE.md) - System architecture

## 🔗 Integration

### Dependencies

- **fks_data**: Real market data (when replacing synthetic endpoints)
- **fks_app**: Business logic integration
- **Ollama** (Optional): Local LLM for development

### Consumers

- **fks_web**: Web interface consumes API endpoints
- **External Clients**: REST API for trading operations
- **WebSocket Clients**: Real-time data streaming

## 📊 Monitoring

### Health Check Endpoints

- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /live` - Liveness probe

### Metrics

- Request count and latency
- WebSocket connection count
- Error rates
- Ollama integration status (if enabled)

### Logging

- API request/response logging
- WebSocket connection events
- Error tracking

## 🛠️ Development

### Setup

```bash
# Clone repository
git clone https://github.com/nuniesmith/fks_api.git
cd fks_api

# Install with Poetry
poetry install --no-root

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install .[websocket,security]
```

### Code Structure

```
repo/api/
├── src/
│   ├── fastapi_main.py     # FastAPI application
│   ├── routes/             # API routes
│   └── websocket/          # WebSocket handlers
├── tests/                  # Test suite
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Contributing

1. Follow Python best practices (PEP 8)
2. Write tests for new endpoints
3. Document API changes
4. Update OpenAPI schema

## 📋 Next Steps

- Replace synthetic data endpoints with real data service
- AuthN/Z middleware integration
- Metrics & tracing
- Deeper Celery task modules / scheduling

---

**Repository**: [nuniesmith/fks_api](https://github.com/nuniesmith/fks_api)  
**Docker Image**: `nuniesmith/fks:api-latest`  
**Status**: Active

