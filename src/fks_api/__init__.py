"""Core package marker for the fks_api service.

Poetry's editable install expects either a top-level module named
`fks_api.py` or a package directory `fks_api/`. The project currently
uses a "flat" src layout with service modules (e.g. `fastapi_main.py`)
at the root of `src/`, so the build failed with:

  ModuleOrPackageNotFoundError: No file/folder found for package fks-api

Creating this package directory satisfies the build backend. We keep it
minimal to avoid altering runtime imports. Convenience re-exports can be
added later if desired (e.g., exposing `app`).
"""

from __future__ import annotations

# (Optional) Expose FastAPI app for tools expecting `fks_api:app`.
try:  # pragma: no cover - defensive import
    from fastapi_main import app  # type: ignore
except Exception:  # noqa: BLE001
    app = None  # type: ignore

__all__ = ["app"]
