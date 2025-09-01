"""Local lightweight stand-ins for the optional `shared_python` package.

These stubs allow the API service to run in standalone mode when the shared
monorepo package is not available (e.g., during minimal CI, local quickstart,
 or container builds that omit shared dependencies).
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Any, Dict


def load_config() -> Dict[str, Any]:  # pragma: no cover - trivial
    return {
        "environment": os.getenv("APP_ENV", "development"),
        "service": os.getenv("API_SERVICE_NAME", "fks_api"),
    }


class StubSettings(dict):  # simple mapping behaving like settings object
    @property
    def environment(self) -> str:  # type: ignore
        return self.get("environment", "development")


@lru_cache
def get_settings() -> StubSettings:
    cfg = load_config()
    return StubSettings(cfg)


def get_risk_threshold() -> float:  # example helper parity with shared
    return float(os.getenv("RISK_THRESHOLD", "0.0"))

__all__ = [
    "load_config",
    "get_settings",
    "get_risk_threshold",
]
