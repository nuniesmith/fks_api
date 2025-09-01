"""Minimal FastAPI + optional Celery bootstrap used by other modules.

The full application (with many routers) is exposed through `fastapi_main.py`.
This module keeps a very small surface for health checks and worker imports.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings

settings = get_settings()

try:  # Celery optional
    from celery import Celery  # type: ignore
except Exception:  # pragma: no cover
    Celery = None  # type: ignore


if TYPE_CHECKING:  # pragma: no cover
    from celery import Celery as CeleryType  # noqa


def _init_celery() -> Optional['CeleryType']:
    broker = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL")
    if Celery is None or not broker:
        return None
    backend = os.getenv("CELERY_RESULT_BACKEND", broker)
    c = Celery("fks_api", broker=broker, backend=backend)
    c.conf.update(task_track_started=True, result_expires=3600)
    return c


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {"status": "healthy", "env": settings.environment, "ts": datetime.now(timezone.utc).isoformat()}

    return app


app = create_app()
celery = _init_celery()

__all__ = ["app", "create_app", "celery"]
