# Using shared_docker Templates with fks_api

This guide shows how to use the shared_docker templates with your fks_api service.

## Benefits of Using Shared Templates

- **🚀 Faster builds**: Better layer caching and optimized build process
- **📦 Standardized**: Consistent Docker setup across all services  
- **🔧 Smart detection**: Auto-detects FastAPI apps and Python modules
- **🐍 Python-optimized**: Built specifically for Python services
- **🏥 Health checks**: Built-in health monitoring
- **🔒 Security**: Non-root user execution

## Quick Start

### Option 1: Use the Build Script (Recommended)
```bash
# Build with shared template
./build.sh

# Build for production
./build.sh --tag fks_api:prod

# Build with GPU support
./build.sh --gpu

# Build without cache
./build.sh --no-cache
```

### Option 2: Use Docker Compose with Shared Template
```bash
# Start with shared template
docker-compose -f docker-compose.shared.yml up --build

# Or override the default
docker-compose -f docker-compose.shared.yml -f docker-compose.override.yml up
```

### Option 3: Direct Docker Build
```bash
# Build using shared Python template directly
docker build -f ../shared/docker/templates/python.Dockerfile -t fks_api:latest .

# With build args
docker build -f ../shared/docker/templates/python.Dockerfile \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg BUILD_PACKAGES="build-essential gcc libpq-dev" \
  -t fks_api:latest .
```

## Environment Variables

The shared template recognizes these environment variables:

### Runtime Configuration
- `APP_MODULE`: Python module to run (default: auto-detected)
- `APP_COMMAND`: Custom command to run instead of auto-detection
- `SERVICE_PORT`: Port to listen on (default: 8000)

### Build Configuration
- `PYTHON_VERSION`: Python version to use (default: 3.11)
- `BUILD_PACKAGES`: Additional system packages to install

### Example Environment
```bash
# .env file
APP_ENV=development
APP_MODULE=fastapi_main:app
SERVICE_PORT=8000
APP_LOG_LEVEL=INFO
```

## File Structure Expected by Template

```
fks_api/
├── requirements.txt          # Production dependencies
├── requirements.dev.txt      # Development dependencies (optional)
├── pyproject.toml           # Python project configuration
└── src/                     # Python source code
    ├── __init__.py
    ├── fastapi_main.py      # Your FastAPI app
    └── ...
```

## Comparison: Current vs Shared Template

| Feature | Current Dockerfile | Shared Template |
|---------|-------------------|-----------------|
| Lines of code | ~800+ | ~50 |
| Build time | Slow (complex multi-stage) | Fast (optimized) |
| Caching | Limited | Excellent |
| Maintenance | High | Low |
| GPU support | Complex setup | Simple `--gpu` flag |
| Auto-detection | Manual configuration | Automatic |

## Migration Steps

1. **Run the migration helper**:
   ```bash
   ./migrate-to-shared.sh
   ```

2. **Test the shared template**:
   ```bash
   docker-compose -f docker-compose.shared.yml up --build
   ```

3. **Verify your app runs correctly**:
   ```bash
   curl http://localhost:8000/health
   ```

4. **Switch to shared template** (when ready):
   ```bash
   mv docker-compose.yml docker-compose.original.yml
   mv docker-compose.shared.yml docker-compose.yml
   ```

## Advanced Usage

### Custom Build Arguments
```bash
./build.sh --arg PYTHON_VERSION=3.12 --arg BUILD_PACKAGES="build-essential gcc libpq-dev redis-tools"
```

### Multiple Services
```yaml
# docker-compose.yml
services:
  fks_api:
    build:
      context: .
      dockerfile: ../shared/docker/templates/python.Dockerfile
  
  fks_worker:
    build:
      context: ../fks_worker
      dockerfile: ../shared/docker/templates/python.Dockerfile
      args:
        APP_COMMAND: "python -m celery worker -A tasks"
```

### Production Deployment
```bash
# Build production image
./build.sh --tag fks_api:v1.0.0

# Deploy with production compose
docker-compose -f docker-compose.prod.yml up -d
```

## Troubleshooting

### Build Issues
- Ensure `requirements.txt` is in the root directory
- Check that `src/` contains your Python code
- Verify shared_docker is available in your workspace

### Runtime Issues
- Check logs: `docker-compose logs fks_api`
- Verify environment variables are set correctly
- Ensure your FastAPI app is in the expected location

### Performance
- The shared template includes optimized layer caching
- Use `.dockerignore` to exclude unnecessary files
- Consider multi-stage builds for production

## Support

For issues with the shared templates, check:
1. `shared/docker/README.md`
2. Template documentation in `shared/docker/templates/`
3. Example configurations in `shared/docker/compose/`
