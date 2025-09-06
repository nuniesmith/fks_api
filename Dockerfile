# FKS API Service - Clean Python Dockerfile
ARG PYTHON_VERSION=3.13
ARG BUILD_TYPE=cpu
ARG CUDA_VERSION=12.8.0
ARG SERVICE_PORT=8000

# Build stage
FROM python:${PYTHON_VERSION}-slim AS build

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    pkg-config \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy and install Python dependencies
ARG INCLUDE_FKS_DATA=0
COPY requirements*.txt ./
RUN set -e; \
        pip install --no-cache-dir --upgrade pip wheel && \
        pip install --no-cache-dir -r requirements.txt && \
        if [ "$INCLUDE_FKS_DATA" = "1" ]; then \
                echo "Including local fks_data package"; \
                if [ -d ../fks_data ]; then \
                    pip install --no-cache-dir -e ../fks_data || echo "WARN: failed editable install of ../fks_data"; \
                else \
                    echo "WARN: ../fks_data not present in build context"; \
                fi; \
        fi

# Copy source code
COPY src/ ./src/

# Smoke test import to fail fast if dependencies or modules are broken
RUN python -c "import sys;print('Python',sys.version);import fastapi_main;print('Imported fastapi_main OK')" || (echo 'Smoke test failed' && exit 1)

# Runtime stage
FROM python:${PYTHON_VERSION}-slim AS final

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy from build stage
COPY --from=build /venv /venv
COPY --from=build /app/src ./src

# Create non-root user
ARG USER_ID=1088
RUN useradd --uid ${USER_ID} --create-home --shell /bin/bash appuser
USER appuser

# Environment variables
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PYTHONPATH=/app/src \
    SERVICE_NAME=fks-api \
    SERVICE_TYPE=api

EXPOSE ${SERVICE_PORT}

ENTRYPOINT ["uvicorn"]

# Copy source code
COPY src/ ./src/

# Service-specific health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

# Default command (can be overridden)
CMD ["src.fastapi_main:app", "--host", "0.0.0.0", "--port", "8000"]
