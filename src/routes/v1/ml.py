"""
ML Inference Endpoints

Provides REST API endpoints for machine learning model predictions.
Supports real-time inference for trading decisions with caching and optimization.
"""

import hashlib
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import httpx
import mlflow
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml"])

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Default MLflow tracking URI
TRAINING_SERVICE_URL = "http://fks-training:8005"  # fks_training service URL

# Cache configuration
CACHE_TTL = 300  # 5 minutes cache TTL for predictions
PREDICTION_CACHE_PREFIX = "ml:prediction:"

# Try to import cache if available
try:
    from framework.cache.cache import Cache
    _cache_available = True
except ImportError:
    _cache_available = False
    logger.warning("Cache framework not available, predictions will not be cached")


class PredictionRequest(BaseModel):
    """Request model for ML predictions"""

    symbol: str = Field(..., description="Trading symbol (e.g., BTC/USDT)")
    sequences: List[List[float]] = Field(
        ..., description="Input sequences of shape (n_samples, sequence_length, n_features)"
    )
    model_name: Optional[str] = Field(
        default="lstm_price_forecast", description="Name of the model to use"
    )
    model_version: Optional[str] = Field(
        default="latest", description="Model version or stage (latest, staging, production)"
    )
    include_sentiment: Optional[bool] = Field(
        default=False, description="Whether to include sentiment analysis in response"
    )


class PredictionResponse(BaseModel):
    """Response model for ML predictions"""

    predictions: List[float] = Field(..., description="Model predictions")
    confidence_intervals: Optional[List[Dict[str, float]]] = Field(
        default=None, description="Confidence intervals for predictions"
    )
    model_info: Dict[str, str] = Field(..., description="Information about the model used")
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None, description="Feature importance scores"
    )


class ExplainRequest(BaseModel):
    """Request model for model explanations"""

    symbol: str = Field(..., description="Trading symbol")
    sequences: List[List[float]] = Field(..., description="Input sequences")
    model_name: Optional[str] = Field(default="lstm_price_forecast")
    model_version: Optional[str] = Field(default="latest")


class ExplainResponse(BaseModel):
    """Response model for explanations"""

    feature_contributions: Dict[str, List[float]] = Field(
        ..., description="Feature contribution scores per sample"
    )
    shap_values: Optional[List[List[float]]] = Field(
        default=None, description="SHAP values for each feature"
    )
    model_info: Dict[str, str] = Field(..., description="Information about the model used")


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make predictions using ML models with caching and optimization.

    This endpoint accepts market data sequences and returns model predictions
    for price forecasting or trading signals. Supports caching for identical inputs.

    Args:
        request: Prediction request with sequences and model information

    Returns:
        PredictionResponse with predictions and metadata
    """
    start_time = time.time()
    
    try:
        # Convert sequences to numpy array
        sequences_array = np.array(request.sequences)

        # Validate input shape
        if len(sequences_array.shape) != 3:
            raise HTTPException(
                status_code=400,
                detail="Sequences must be 3D array: (n_samples, sequence_length, n_features)",
            )

        # Generate cache key
        cache_key = _generate_cache_key(request)
        
        # Try to get from cache
        if _cache_available:
            try:
                from framework.cache.cache import get_cache
                cache = get_cache()
                cached_result = await cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for prediction: {cache_key[:20]}...")
                    cached_data = json.loads(cached_result)
                    # Add cache info to response
                    cached_data["model_info"]["cached"] = True
                    return PredictionResponse(**cached_data)
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")

        # Load model from MLflow
        model_uri = await _get_model_uri(request.model_name, request.model_version)
        model = mlflow.pyfunc.load_model(model_uri)

        # Make predictions
        predictions = model.predict(sequences_array)

        # Convert to list
        predictions_list = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions

        # Calculate inference latency
        inference_time = time.time() - start_time

        # Calculate confidence intervals using prediction variance
        confidence_intervals = _calculate_confidence_intervals(
            model, sequences_array, predictions_list
        )

        # Extract feature importance using SHAP if available
        feature_importance = _extract_feature_importance(
            model, sequences_array, request.model_name
        )

        # Get sentiment analysis if requested
        sentiment_data = None
        if request.include_sentiment:
            try:
                sentiment_data = await _get_sentiment_analysis(request.symbol)
            except Exception as e:
                logger.warning(f"Failed to get sentiment analysis: {e}")

        # Get model info
        model_info = {
            "model_name": request.model_name,
            "model_version": request.model_version,
            "model_uri": model_uri,
            "cached": False,
            "inference_time_ms": round(inference_time * 1000, 2),
        }

        response = PredictionResponse(
            predictions=predictions_list,
            confidence_intervals=confidence_intervals,
            model_info=model_info,
            feature_importance=feature_importance,
        )

        # Add sentiment to response if available
        if sentiment_data:
            response.model_info["sentiment"] = sentiment_data

        # Cache the result
        if _cache_available:
            try:
                from framework.cache.cache import get_cache
                cache = get_cache()
                await cache.set(
                    cache_key,
                    json.dumps(response.dict()),
                    ttl=CACHE_TTL,
                )
                logger.debug(f"Cached prediction: {cache_key[:20]}...")
            except Exception as e:
                logger.warning(f"Cache store failed: {e}")

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest) -> ExplainResponse:
    """
    Explain model predictions using SHAP.

    This endpoint provides feature importance and contribution scores
    for model predictions, enabling interpretability and transparency.

    Args:
        request: Explanation request with sequences and model information

    Returns:
        ExplainResponse with feature contributions and SHAP values
    """
    try:
        # Convert sequences to numpy array
        sequences_array = np.array(request.sequences)

        # Load model from MLflow
        model_uri = await _get_model_uri(request.model_name, request.model_version)
        model = mlflow.pyfunc.load_model(model_uri)

        # Try to use SHAP for explanations
        try:
            import shap

            # Create SHAP explainer
            # Use a subset of data as background for TreeExplainer or KernelExplainer
            background_data = sequences_array[: min(100, len(sequences_array))]

            # Try TreeExplainer first (for tree-based models), fallback to KernelExplainer
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                # Fallback to KernelExplainer for non-tree models
                explainer = shap.KernelExplainer(
                    model.predict, background_data[:10]
                )  # Use smaller background for speed

            # Calculate SHAP values
            shap_values = explainer.shap_values(sequences_array)

            # Convert to list format
            if isinstance(shap_values, list):
                shap_values_list = [sv.tolist() if isinstance(sv, np.ndarray) else sv for sv in shap_values]
            else:
                shap_values_list = shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values

            # Calculate feature contributions (mean absolute SHAP values)
            if isinstance(shap_values, np.ndarray):
                mean_contributions = np.abs(shap_values).mean(axis=0)
                feature_contributions = {
                    f"feature_{i}": mean_contributions[i].tolist()
                    if isinstance(mean_contributions[i], np.ndarray)
                    else [float(mean_contributions[i])]
                    for i in range(len(mean_contributions))
                }
            else:
                # For multi-output models
                feature_contributions = {
                    f"feature_{i}": [0.0] * len(sequences_array)
                    for i in range(sequences_array.shape[2] if len(sequences_array.shape) > 2 else 1)
                }

        except ImportError:
            logger.warning("SHAP not available, returning placeholder explanations")
            # Fallback: return placeholder
            n_samples = len(sequences_array)
            n_features = sequences_array.shape[2] if len(sequences_array.shape) > 2 else 1

            feature_contributions = {
                f"feature_{i}": [0.0] * n_samples for i in range(n_features)
            }
            shap_values_list = None

        model_info = {
            "model_name": request.model_name,
            "model_version": request.model_version,
            "model_uri": model_uri,
            "explainer": "SHAP" if "shap" in sys.modules else "placeholder",
        }

        return ExplainResponse(
            feature_contributions=feature_contributions,
            shap_values=shap_values_list,
            model_info=model_info,
        )

    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.get("/models")
async def list_models(
    stage: Optional[str] = Query(None, description="Filter by model stage (staging, production)"),
) -> Dict:
    """
    List available ML models from MLflow registry.

    Args:
        stage: Optional filter by model stage

    Returns:
        Dictionary with list of available models
    """
    try:
        # Connect to MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        # Search for registered models
        if stage:
            models = client.search_registered_models(filter_string=f"tags.stage='{stage}'")
        else:
            models = client.search_registered_models()

        models_list = []
        for model in models:
            latest_version = model.latest_versions[0] if model.latest_versions else None
            models_list.append(
                {
                    "name": model.name,
                    "latest_version": latest_version.version if latest_version else None,
                    "stages": [v.stage for v in model.latest_versions],
                    "description": model.description,
                }
            )

        return {"models": models_list, "count": len(models_list)}

    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/{model_name}/versions")
async def list_model_versions(model_name: str) -> Dict:
    """
    List all versions of a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with list of model versions
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        versions = client.search_model_versions(f"name='{model_name}'")

        versions_list = []
        for version in versions:
            versions_list.append(
                {
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "created_at": version.creation_timestamp,
                }
            )

        return {"model_name": model_name, "versions": versions_list, "count": len(versions_list)}

    except Exception as e:
        logger.error(f"Failed to list model versions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list model versions: {str(e)}")


def _calculate_confidence_intervals(
    model: Any,
    sequences: np.ndarray,
    predictions: List[float],
    confidence_level: float = 0.95,
) -> Optional[List[Dict[str, float]]]:
    """
    Calculate confidence intervals for predictions.

    Uses Monte Carlo dropout or prediction variance for uncertainty quantification.

    Args:
        model: Trained ML model
        sequences: Input sequences
        predictions: Model predictions
        confidence_level: Confidence level (default: 0.95 for 95% CI)

    Returns:
        List of confidence intervals [{"lower": float, "upper": float}, ...]
    """
    try:
        # Method 1: Try Monte Carlo dropout (if model supports it)
        if hasattr(model, "predict_with_uncertainty"):
            # Model has built-in uncertainty estimation
            uncertainties = model.predict_with_uncertainty(sequences)
            if uncertainties is not None:
                intervals = []
                for pred, unc in zip(predictions, uncertainties):
                    # Assume normal distribution
                    std = unc if isinstance(unc, (int, float)) else unc.get("std", 0.0)
                    z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
                    margin = z_score * std
                    intervals.append({
                        "lower": float(pred - margin),
                        "upper": float(pred + margin),
                        "confidence": confidence_level,
                    })
                return intervals

        # Method 2: Use prediction variance from multiple forward passes (Monte Carlo)
        # This requires model to support dropout at inference time
        try:
            # Try to get model's underlying PyTorch model if it's wrapped
            pytorch_model = None
            if hasattr(model, "unwrap_python_model"):
                pytorch_model = model.unwrap_python_model()
            elif hasattr(model, "_model"):
                pytorch_model = model._model

            if pytorch_model is not None:
                import torch

                # Enable dropout for uncertainty estimation
                pytorch_model.train()  # Enable dropout
                n_samples = 10  # Number of Monte Carlo samples

                mc_predictions = []
                with torch.no_grad():
                    sequences_tensor = torch.FloatTensor(sequences)
                    for _ in range(n_samples):
                        pred = pytorch_model(sequences_tensor)
                        mc_predictions.append(pred.numpy())

                # Calculate variance across samples
                mc_predictions = np.array(mc_predictions)
                pred_std = np.std(mc_predictions, axis=0).flatten()

                # Calculate confidence intervals
                z_score = 1.96 if confidence_level == 0.95 else 2.576
                intervals = []
                for pred, std in zip(predictions, pred_std):
                    margin = z_score * std
                    intervals.append({
                        "lower": float(pred - margin),
                        "upper": float(pred + margin),
                        "confidence": confidence_level,
                        "std": float(std),
                    })

                # Set model back to eval mode
                pytorch_model.eval()
                return intervals

        except Exception as e:
            logger.debug(f"Monte Carlo dropout failed: {e}")

        # Method 3: Use empirical prediction error (if we have historical errors)
        # For now, use a simple heuristic based on prediction magnitude
        intervals = []
        for pred in predictions:
            # Simple heuristic: 2% of prediction value as uncertainty
            margin = abs(pred) * 0.02
            intervals.append({
                "lower": float(pred - margin),
                "upper": float(pred + margin),
                "confidence": confidence_level,
                "method": "heuristic",
            })

        return intervals

    except Exception as e:
        logger.warning(f"Failed to calculate confidence intervals: {e}")
        return None


def _extract_feature_importance(
    model: Any, sequences: np.ndarray, model_name: str
) -> Optional[Dict[str, float]]:
    """
    Extract feature importance from model predictions.

    Uses SHAP if available, otherwise falls back to permutation importance.

    Args:
        model: Trained ML model
        sequences: Input sequences
        model_name: Name of the model

    Returns:
        Dictionary mapping feature names to importance scores
    """
    try:
        # Try SHAP first
        try:
            import shap

            # Create explainer
            background_data = sequences[: min(100, len(sequences))]

            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                # Fallback to KernelExplainer
                explainer = shap.KernelExplainer(
                    model.predict, background_data[:10]
                )

            # Calculate SHAP values
            shap_values = explainer.shap_values(sequences)

            # Calculate mean absolute SHAP values per feature
            if isinstance(shap_values, np.ndarray):
                feature_importance = np.abs(shap_values).mean(axis=0)
            elif isinstance(shap_values, list):
                # Multi-output model
                feature_importance = np.abs(np.array(shap_values)).mean(axis=(0, 1))
            else:
                feature_importance = np.abs(shap_values).mean(axis=0)

            # Create feature names
            n_features = sequences.shape[2] if len(sequences.shape) > 2 else len(feature_importance)
            importance_dict = {
                f"feature_{i}": float(importance)
                for i, importance in enumerate(feature_importance[:n_features])
            }

            return importance_dict

        except ImportError:
            logger.debug("SHAP not available, using permutation importance")

        # Fallback: Permutation importance
        # This is computationally expensive, so we'll use a simplified version
        baseline_predictions = model.predict(sequences)
        baseline_error = np.mean(np.abs(baseline_predictions))

        n_features = sequences.shape[2] if len(sequences.shape) > 2 else sequences.shape[1]
        importance_scores = {}

        # Sample a subset for efficiency
        sample_size = min(10, len(sequences))
        sample_indices = np.random.choice(len(sequences), sample_size, replace=False)
        sample_sequences = sequences[sample_indices]

        for i in range(n_features):
            # Permute feature i
            permuted_sequences = sample_sequences.copy()
            np.random.shuffle(permuted_sequences[:, :, i] if len(permuted_sequences.shape) > 2 else permuted_sequences[:, i])

            # Get predictions with permuted feature
            permuted_predictions = model.predict(permuted_sequences)
            permuted_error = np.mean(np.abs(permuted_predictions))

            # Importance = increase in error when feature is permuted
            importance = permuted_error - baseline_error
            importance_scores[f"feature_{i}"] = float(importance)

        # Normalize importance scores
        max_importance = max(abs(v) for v in importance_scores.values()) if importance_scores else 1.0
        if max_importance > 0:
            importance_scores = {
                k: v / max_importance for k, v in importance_scores.items()
            }

        return importance_scores

    except Exception as e:
        logger.warning(f"Failed to extract feature importance: {e}")
        return None


async def _get_sentiment_analysis(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get sentiment analysis from fks_ai service.

    Args:
        symbol: Trading symbol

    Returns:
        Sentiment analysis dictionary or None
    """
    try:
        ai_service_url = os.getenv("FKS_AI_URL", "http://fks-ai:8007")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{ai_service_url}/ai/analyze",
                json={
                    "symbol": symbol,
                    "market_data": {},
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract sentiment from analysis
            sentiment_analysis = data.get("analysts_output", {}).get("sentiment", "")
            confidence = data.get("confidence", 0.5)

            return {
                "score": _extract_sentiment_score(sentiment_analysis),
                "confidence": confidence,
                "analysis": sentiment_analysis,
            }
    except Exception as e:
        logger.debug(f"Sentiment analysis failed: {e}")
        return None


def _extract_sentiment_score(analysis_text: str) -> float:
    """Extract numeric sentiment score from analysis text."""
    text_lower = analysis_text.lower()
    bullish_keywords = ["bullish", "buy", "positive", "greed", "accumulate"]
    bearish_keywords = ["bearish", "sell", "negative", "fear", "distribute"]

    bullish_count = sum(1 for kw in bullish_keywords if kw in text_lower)
    bearish_count = sum(1 for kw in bearish_keywords if kw in text_lower)

    if bullish_count > bearish_count:
        return min(1.0, 0.5 + (bullish_count * 0.1))
    elif bearish_count > bullish_count:
        return max(-1.0, -0.5 - (bearish_count * 0.1))
    else:
        return 0.0


def _generate_cache_key(request: PredictionRequest) -> str:
    """
    Generate cache key for prediction request.
    
    Args:
        request: Prediction request
        
    Returns:
        Cache key string
    """
    # Create hash of sequences and model info
    key_data = {
        "sequences": request.sequences,
        "model_name": request.model_name,
        "model_version": request.model_version,
    }
    key_str = json.dumps(key_data, sort_keys=True)
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()
    return f"{PREDICTION_CACHE_PREFIX}{key_hash}"


async def _get_model_uri(model_name: str, model_version: str) -> str:
    """
    Get model URI from MLflow registry.

    Args:
        model_name: Name of the model
        model_version: Version or stage (latest, staging, production)

    Returns:
        Model URI for loading
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    if model_version in ["latest", "staging", "production", "archive", "None"]:
        # Get model by stage
        try:
            model_version_obj = client.get_model_version(model_name, model_version)
            return f"models:/{model_name}/{model_version}"
        except Exception:
            # Fallback to latest version
            model_version_obj = client.get_latest_versions(model_name, stages=[model_version])[0]
            return f"models:/{model_name}/{model_version_obj.version}"
    else:
        # Get specific version
        return f"models:/{model_name}/{model_version}"

