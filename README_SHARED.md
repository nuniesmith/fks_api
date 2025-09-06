# FKS API Service

## Overview

Lightweight FastAPI service providing HTTP/WebSocket endpoints for the FKS platform. This service now uses **shared Docker templates and scripts** via Git submodules for standardized deployment across all FKS services.

## 🚀 Quick Start with Shared Resources

### Prerequisites
- Docker & Docker Compose
- Git
- Python 3.11+ (for local development)

### 1. Initialize Shared Resources
```bash
# Initialize shared templates and scripts
./setup-shared_resources.sh

# Or manually:
git submodule update --init --recursive
```

### 2. Start the Service
```bash
# Using the enhanced start script
./start.sh

# Or with options
./start.sh --dev --logs

# Or direct docker-compose
docker-compose up --build
```

### 3. Test the Service
```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

## Features

- Health & info endpoints
- Synthetic chart + indicator data for UI development
- Optional background Ollama model pull (dev)
- Modular router loading with graceful degradation
- **Shared Docker templates** for consistent deployment
- **Shared entrypoint scripts** with smart service detection

## 📁 Project Structure with Shared Resources

```
fks_api/
├── src/                          # Python source code
│   ├── fastapi_main.py          # FastAPI application entry point
│   ├── routers/                 # API route definitions
│   ├── middleware/              # Custom middleware
│   └── services/                # Business logic services
├── shared/                       # Git submodules (shared resources)
│   ├── shared_docker/           # Docker templates & compose files
│   │   ├── templates/           # Dockerfile templates
│   │   └── scripts/             # Build scripts
│   └── shared_scripts/          # Shared shell scripts
│       └── docker/              # Docker entrypoint scripts
├── docker-compose.yml           # Uses shared templates
├── build.sh                     # Uses shared build system
├── start.sh                     # Enhanced startup with submodule support
├── entrypoint.sh                # Delegates to shared entrypoint
└── setup-shared_resources.sh    # Initialize shared resources
```

## 🛠️ Development

### Local Development (Poetry - Recommended)
```bash
# Poetry setup
poetry install --no-root
poetry run uvicorn fks_api.fastapi_main:app --reload --port 8000

# Or plain venv + pip
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install .[websocket,security]
uvicorn fks_api.fastapi_main:app --reload --port 8000
```

### Building with Shared Templates
```bash
# Build with shared templates (recommended)
./build.sh

# Build for production
./build.sh --tag fks_api:prod

# Build with GPU support
./build.sh --gpu
```

## 🐳 Docker Usage with Shared Resources

### Standard Commands
```bash
# Build and start with shared templates
docker-compose up --build

# Development mode with shared resources
./start.sh --dev

# Follow logs
./start.sh --logs

# Production mode
./start.sh --prod
```

### Legacy Docker Commands (still supported)
```bash
# Dev (build + live reload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Production (pre-built images)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Pull-only sanity
docker compose -f docker-compose.yml -f docker-compose.pull-only.yml pull
```

## 🔧 Configuration

### Environment Variables
- `SERVICE_TYPE=api` - Service type for shared scripts
- `SERVICE_PORT=8000` - Port to listen on
- `APP_MODULE=fastapi_main:app` - FastAPI module to run
- `APP_ENV=development` - Environment mode
- `API_SERVICE_NAME`, `API_SERVICE_PORT` - Legacy API config
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_FAST_MODEL` - Ollama integration

### Shared Resources Configuration
This service uses two shared repositories as Git submodules:
- **shared_docker**: Docker templates, compose files, build scripts
- **shared_scripts**: Shell scripts for entrypoints, health checks, utilities

## 🔄 Shared Resource Management

### Update Shared Resources
```bash
# Update to latest versions
./setup-shared_resources.sh --update

# Force reinitialize (if there are issues)
./setup-shared_resources.sh --force-reinit

# Test the setup
./setup-shared_resources.sh --test
```

### Troubleshooting Shared Resources
```bash
# Check submodule status
git submodule status

# Manual submodule update
git submodule update --init --recursive

# Initialize submodules only
./start.sh --init-submodules
```

## 🧪 Testing

### Smoke Test
```bash
poetry run pytest -q
```

### Test with Shared Resources
```bash
# Test the shared resource setup
./setup-shared_resources.sh --test

# Health check via shared scripts
curl http://localhost:8000/health
```

## 📊 API Endpoints

Visit: <http://localhost:8000/docs> for interactive documentation

Key endpoints:
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `GET /info` - Service information
- WebSocket endpoints for real-time data

## 🎯 Celery Integration

```bash
docker compose exec worker celery -A app.celery call fks_api.ping
```

## ✅ Benefits of Shared Resources

1. **🔄 Consistency**: Same Docker setup across all FKS services
2. **🚀 Fast builds**: Optimized caching and build processes  
3. **🛠️ Easy maintenance**: Updates apply to all services
4. **📊 Standardized monitoring**: Consistent health checks and logging
5. **🔧 Smart automation**: Auto-detection and fallback strategies
6. **📦 Version control**: Shared resources are version-controlled via submodules

## 🚀 Production Deployment

### Production Build
```bash
# Build production image
./build.sh --tag fks_api:v1.0.0

# Start with production config
./start.sh --prod
```

## 📋 Next Steps

- Replace synthetic data endpoints with real data service
- AuthN/Z middleware integration with shared security templates
- Metrics & tracing using shared monitoring scripts
- Deeper Celery task modules / scheduling
- **Migrate other FKS services** to use shared resources

## 📚 Related Documentation

- [Shared Docker Templates](shared/shared_docker/README.md)
- [Shared Scripts Documentation](shared/shared_scripts/README.md)
- [Using Shared Docker Guide](SHARED_DOCKER_GUIDE.md)
