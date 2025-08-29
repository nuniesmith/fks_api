#!/usr/bin/env bash
set -euo pipefail

# migrate-to-shared.sh
# Helper script to migrate fks_api to use shared_docker templates

echo "🚀 Migrating fks_api to shared_docker templates..."

SCRIPT_DIR=$(dirname "$0")
WORKSPACE_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || dirname "$SCRIPT_DIR")

# Backup original Dockerfile
if [[ -f "$SCRIPT_DIR/Dockerfile" && ! -f "$SCRIPT_DIR/Dockerfile.backup" ]]; then
    echo "📦 Backing up original Dockerfile..."
    cp "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR/Dockerfile.backup"
    echo "   ✅ Saved as Dockerfile.backup"
fi

# Check if shared_docker is available
SHARED_DOCKERFILE="$WORKSPACE_ROOT/shared/shared_docker/templates/python.Dockerfile"
if [[ ! -f "$SHARED_DOCKERFILE" ]]; then
    echo "❌ shared_docker templates not found at $SHARED_DOCKERFILE"
    echo "   Please ensure shared_docker is properly set up in your workspace"
    exit 1
fi

echo "✅ Found shared_docker templates"

# Create a simplified Dockerfile that references the shared template
cat > "$SCRIPT_DIR/Dockerfile.shared" << 'EOF'
# Simplified Dockerfile using shared_docker template
# For more advanced builds, use: docker build -f ../shared/shared_docker/templates/python.Dockerfile .

# Include the shared Python template
FROM python:3.11-slim AS base

# Copy requirements to leverage Docker layer caching
COPY requirements*.txt ./
COPY pyproject.toml ./

# Use the shared template build logic
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip wheel setuptools && \
    /opt/venv/bin/pip install -r requirements.txt

# Copy application source
COPY src ./src

ENV PYTHONPATH=/app/src:/app \
    PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

# Smart entrypoint that auto-detects FastAPI apps
ENV APP_MODULE="${APP_MODULE:-app.main:app}"
CMD ["bash", "-c", "if [ -n \"$APP_COMMAND\" ]; then exec $APP_COMMAND; elif command -v uvicorn >/dev/null 2>&1; then exec uvicorn $APP_MODULE --host 0.0.0.0 --port ${SERVICE_PORT:-8000}; else exec bash; fi"]
EOF

echo "📝 Created Dockerfile.shared (simplified version)"

# Update .dockerignore if needed
if [[ ! -f "$SCRIPT_DIR/.dockerignore" ]]; then
    cat > "$SCRIPT_DIR/.dockerignore" << 'EOF'
.git
.env
*.md
.pytest_cache
__pycache__
.venv
*.backup
node_modules
.DS_Store
EOF
    echo "📝 Created .dockerignore"
fi

echo ""
echo "🎉 Migration setup complete!"
echo ""
echo "Next steps:"
echo "1. Test with shared template:"
echo "   docker-compose -f docker-compose.shared.yml up --build"
echo ""
echo "2. Or use the build script:"
echo "   ./build.sh"
echo ""
echo "3. For production builds:"
echo "   ./build.sh --tag fks_api:prod"
echo ""
echo "4. For GPU builds:"
echo "   ./build.sh --gpu --tag fks_api:gpu"
echo ""
echo "Files created:"
echo "  ✅ build.sh (executable build script)"
echo "  ✅ docker-compose.shared.yml (using shared template)"  
echo "  ✅ Dockerfile.shared (simplified reference)"
echo "  ✅ Dockerfile.backup (backup of original)"
echo ""
echo "The shared template provides:"
echo "  🚀 Faster builds with better caching"
echo "  🐍 Auto-detection of Python/FastAPI apps"
echo "  🔧 Standardized environment setup"
echo "  📦 Smart dependency management"
echo "  🏥 Built-in health checks"
