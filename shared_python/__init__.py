"""Local stub of shared_python for simplified fks_api build.

The full monorepo provides this as a path dependency outside this service
directory. To keep Dockerfile.simple self-contained we expose the symbols
tests expect. Safe to replace with the real package when available.
"""
from __future__ import annotations
from typing import Any, Dict

_SETTINGS: Dict[str, Any] = {"environment": "development", "version": "stub"}


def load_config(*_a: Any, **_k: Any) -> Dict[str, Any]:  # pragma: no cover
    return _SETTINGS


def get_settings(*_a: Any, **_k: Any) -> Dict[str, Any]:  # pragma: no cover
    return _SETTINGS


def get_risk_threshold(*_a: Any, **_k: Any) -> float:  # pragma: no cover
    return 0.0
