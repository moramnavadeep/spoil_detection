# REST API Reference Manual

Interactive OpenAPI documentation is hosted locally at: `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

---

## 1. Health Check
Checks service uptime and model readiness.

- **Method**: `GET`
- **Path**: `/health` (or `/api/v1/health`)
- **Response**: `200 OK`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "environment": "development"
}
```

---

## 2. Model Metadata
Returns runtime metadata, input dimensions, and target categories.

- **Method**: `GET`
- **Path**: `/model/metadata` (or `/api/v1/model/metadata`)
- **Response**: `200 OK`

```json
{
  "model_name": "NutriFresh AI - Food Spoilage Detection",
  "version": "1.0.0",
  "architecture": "Convolutional Neural Network (CNN - 3 Blocks)",
  "input_shape": [128, 128, 3],
  "classes": ["Fresh", "Rotten"],
  "total_parameters": 1033282,
  "model_format": ".h5"
}
```

---

## 3. Predict Single Image
Analyzes a single uploaded food image for freshness.

- **Method**: `POST`
- **Path**: `/predict` (or `/api/v1/predict`)
- **Content-Type**: `multipart/form-data`
- **Parameters**: `file` (binary image data)

### Example `cURL` Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@samples/fresh_apple.jpg"
```

### Example Python Client:
```python
import requests

with open("samples/fresh_apple.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/predict", files={"file": f})

print(response.json())
```

### Success Response (`200 OK`):
```json
{
  "filename": "fresh_apple.jpg",
  "prediction": "Fresh",
  "is_fresh": true,
  "confidence": 0.9751,
  "confidence_percentage": 97.51,
  "probabilities": {
    "Fresh": 0.9751,
    "Rotten": 0.0249
  },
  "latency_ms": 14.82,
  "model_version": "1.0.0"
}
```

---

## 4. Predict Batch Images
Analyzes multiple food images in a single batch request.

- **Method**: `POST`
- **Path**: `/predict/batch` (or `/api/v1/predict/batch`)
- **Content-Type**: `multipart/form-data`
- **Parameters**: `files` (repeated image binary files)

### Example `cURL` Request:
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@samples/fresh_apple.jpg" \
     -F "files=@samples/rotten_apple.webp"
```

### Success Response (`200 OK`):
```json
{
  "total_items": 2,
  "successful_items": 2,
  "failed_items": 0,
  "predictions": [
    {
      "filename": "fresh_apple.jpg",
      "success": true,
      "result": {
        "filename": "fresh_apple.jpg",
        "prediction": "Fresh",
        "is_fresh": true,
        "confidence": 0.9751,
        "confidence_percentage": 97.51,
        "probabilities": {
          "Fresh": 0.9751,
          "Rotten": 0.0249
        },
        "latency_ms": 13.91,
        "model_version": "1.0.0"
      },
      "error": null
    },
    {
      "filename": "rotten_apple.webp",
      "success": true,
      "result": {
        "filename": "rotten_apple.webp",
        "prediction": "Rotten",
        "is_fresh": false,
        "confidence": 0.6189,
        "confidence_percentage": 61.89,
        "probabilities": {
          "Fresh": 0.3811,
          "Rotten": 0.6189
        },
        "latency_ms": 12.44,
        "model_version": "1.0.0"
      },
      "error": null
    }
  ]
}
```
