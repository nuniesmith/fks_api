"""Lightweight Optimization API router to back Strategy Optimization modal."""
from __future__ import annotations

try:  # local import after future
    from services.api.services.data_service import DataService  # type: ignore
except Exception:  # pragma: no cover
    from services.data_service import DataService

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel

from framework.middleware.auth import authenticate_user, get_auth_token

router = APIRouter(prefix="/optimize", tags=["optimize"])


class OptimizationInput(BaseModel):
    code: str
    language: str
    parameters: dict[str, Any]


class OptimizationResult(BaseModel):
    parameters: dict[str, Any]
    trials: int
    bestScore: float


@router.post("")
async def optimize(input: OptimizationInput, token: str = Depends(get_auth_token)) -> OptimizationResult:
    """
    Mock optimization endpoint: nudges numeric parameters by +5%.
    Replace with real Optuna-based study when available.
    """
    authenticate_user(token)

    try:
        params = dict(input.parameters)
        for k, v in list(params.items()):
            if isinstance(v, (int, float)):
                params[k] = round(float(v) * 1.05, 6)
        # Fake score
        best_score = 1.0
        return OptimizationResult(parameters=params, trials=30, bestScore=best_score)
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")
