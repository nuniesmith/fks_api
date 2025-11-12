# ML API Guide

Complete guide to the ML inference endpoints in `fks_api`.

## Endpoints

### POST `/api/v1/ml/predict`

Make predictions using trained ML models.

**Request:**
```json
{
  "symbol": "BTC/USDT",
  "sequences": [[1.0, 2.0, 3.0, 4.0, 5.0], ...],
  "model_name": "lstm_price_forecast",
  "model_version": "latest",
  "include_sentiment": false
}
```

**Response:**
```json
{
  "predictions": [100.5, 101.2, ...],
  "confidence_intervals": [
    {
      "lower": 99.0,
      "upper": 102.0,
      "confidence": 0.95,
      "std": 0.5
    }
  ],
  "model_info": {
    "model_name": "lstm_price_forecast",
    "model_version": "latest",
    "model_uri": "models:/lstm_price_forecast/1",
    "cached": false,
    "inference_time_ms": 45.2
  },
  "feature_importance": {
    "feature_0": 0.15,
    "feature_1": 0.23,
    ...
  }
}
```

### POST `/api/v1/ml/explain`

Get model explanations using SHAP.

**Request:**
```json
{
  "symbol": "BTC/USDT",
  "sequences": [[...]],
  "model_name": "lstm_price_forecast",
  "model_version": "latest"
}
```

**Response:**
```json
{
  "feature_contributions": {
    "feature_0": [0.1, 0.2, ...],
    "feature_1": [0.15, 0.25, ...]
  },
  "shap_values": [[0.1, 0.2, ...], ...],
  "model_info": {
    "model_name": "lstm_price_forecast",
    "explainer": "SHAP"
  }
}
```

### GET `/api/v1/ml/models`

List all registered models.

**Query Parameters:**
- `stage` (optional): Filter by stage (staging, production)

**Response:**
```json
{
  "models": [
    {
      "name": "lstm_price_forecast",
      "latest_version": "1",
      "stages": ["staging", "production"],
      "description": "LSTM model for price forecasting"
    }
  ],
  "count": 1
}
```

### GET `/api/v1/ml/models/{model_name}/versions`

Get all versions of a model.

**Response:**
```json
[
  {
    "version": "1",
    "stage": "production",
    "run_id": "...",
    "creation_timestamp": "...",
    "metrics": {...},
    "params": {...}
  }
]
```

## Python Client Example

```python
import httpx

class MLAPIClient:
    def __init__(self, base_url: str = "http://fks-api:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def predict(
        self,
        symbol: str,
        sequences: list,
        model_name: str = "lstm_price_forecast",
        include_sentiment: bool = False,
    ):
        response = await self.client.post(
            f"{self.base_url}/api/v1/ml/predict",
            json={
                "symbol": symbol,
                "sequences": sequences,
                "model_name": model_name,
                "include_sentiment": include_sentiment,
            },
        )
        return response.json()

    async def explain(self, symbol: str, sequences: list):
        response = await self.client.post(
            f"{self.base_url}/api/v1/ml/explain",
            json={
                "symbol": symbol,
                "sequences": sequences,
            },
        )
        return response.json()

# Usage
client = MLAPIClient()
result = await client.predict("BTC/USDT", sequences)
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Model not found
- `500`: Internal server error

Error response format:
```json
{
  "detail": "Error message"
}
```

## Caching

Predictions are cached for 5 minutes to improve performance. Cache keys are based on:
- Input sequences
- Model name and version
- Symbol

## Rate Limiting

API endpoints respect rate limits configured in the service. Check response headers for rate limit information.

