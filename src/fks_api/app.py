"""App + Celery bootstrap providing a minimal health endpoint and Celery app.

The richer FastAPI application (with many routers) lives in ``fastapi_main.py``.
This module exposes ``celery`` (referenced by docker compose commands) and a
small FastAPI app for lightweight liveness.
"""

from __future__ import annotations

import os
from typing import Optional
from fastapi import FastAPI

try:  # Celery optional for minimal API usage
	from celery import Celery  # type: ignore
except Exception:  # pragma: no cover
	Celery = None  # type: ignore

from fks_shared_python import get_settings

settings = get_settings()


def create_app() -> FastAPI:
	app = FastAPI(title="FKS API (Core)", version="0.1.0")

	@app.get("/health")
	def health() -> dict[str, str]:
		return {"status": "ok", "env": settings.APP_ENV}

	return app


app = create_app()


def _create_celery() -> "Optional[Celery]":
	if Celery is None:
		return None
	broker = os.getenv("REDIS_URL", "redis://redis:6379/0")
	backend = broker
	c = Celery("fks_api", broker=broker, backend=backend)
	c.conf.update(
		task_acks_late=True,
		worker_prefetch_multiplier=int(os.getenv("WORKER_PREFETCH_MULTIPLIER", "1")),
		task_serializer="json",
		accept_content=["json"],
		result_serializer="json",
		timezone=os.getenv("TZ", "UTC"),
		enable_utc=True,
	)

	@c.task(name="fks_api.ping")
	def ping() -> str:  # type: ignore
		return "pong"

	return c


celery = _create_celery()

__all__ = ["app", "create_app", "celery"]


