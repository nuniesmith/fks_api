"""Primary FastAPI application with routers.

`app.py` provides a minimal bootstrap. This module expands the application by
including available routers, while failing gracefully if optional dependencies
are missing. Tests import `fastapi_main:app`.
"""
from __future__ import annotations

from typing import Any

from app import app  # reuse the minimal app instance

# Router inclusion helpers -------------------------------------------------

def _include(prefix: str, import_path: str, attr: str = "router") -> None:
    try:
        module = __import__(import_path, fromlist=[attr])
        router = getattr(module, attr)
        app.include_router(router, prefix=prefix)
    except Exception as e:  # pragma: no cover - best effort
        print(f"[fastapi_main] Skipped {import_path}: {e}")

# Core routers (best-effort)
_include("/api", "routers.active_assets")
_include("/api", "routers.backtests_simple")
_include("/api", "routers.data_quality")
_include("/api", "routers.dataset")
_include("/api", "routers.signals")
_include("/api", "routers.strategies")
_include("/api", "routers.trading_sessions")
_include("/api", "routers.optimization")
_include("/api", "routers.transformer_ingest")

# Legacy v1 routes if available
_include("/api/v1", "routes.v1.backtest")

@app.get("/")
async def root() -> dict[str, Any]:
    return {"service": "fks_api", "status": "ok"}

__all__ = ["app"]
