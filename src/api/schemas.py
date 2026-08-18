"""Pydantic Request and Response Schemas."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    version: str = Field(..., examples=["1.0.0"])
    environment: str = Field(..., examples=["development"])


class ModelMetadataResponse(BaseModel):
    """Model architecture and runtime metadata schema."""
    model_name: str = Field(..., examples=["NutriFresh AI - Food Spoilage Detection"])
    version: str = Field(..., examples=["1.0.0"])
    architecture: str = Field(..., examples=["Convolutional Neural Network (CNN - 3 Blocks)"])
    input_shape: List[int] = Field(..., examples=[[128, 128, 3]])
    classes: List[str] = Field(..., examples=[["Fresh", "Rotten"]])
    total_parameters: int = Field(..., examples=[1033282])
    model_format: str = Field(..., examples=[".h5"])


class SinglePredictionResult(BaseModel):
    """Prediction result for a single image."""
    filename: Optional[str] = Field(None, examples=["apple.jpg"])
    prediction: str = Field(..., examples=["Fresh"])
    is_fresh: bool = Field(..., examples=[True])
    confidence: float = Field(..., examples=[0.9842])
    confidence_percentage: float = Field(..., examples=[98.42])
    probabilities: Dict[str, float] = Field(..., examples=[{"Fresh": 0.9842, "Rotten": 0.0158}])
    latency_ms: float = Field(..., examples=[14.52])
    model_version: str = Field(..., examples=["1.0.0"])


class BatchItemResult(BaseModel):
    """Result wrapper for an item within a batch prediction."""
    filename: str = Field(..., examples=["sample_1.jpg"])
    success: bool = Field(..., examples=[True])
    result: Optional[SinglePredictionResult] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""
    total_items: int = Field(..., examples=[5])
    successful_items: int = Field(..., examples=[5])
    failed_items: int = Field(..., examples=[0])
    predictions: List[BatchItemResult]


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    error: str = Field(..., examples=["Invalid image file format"])
    detail: Optional[str] = Field(None, examples=["Expected JPEG, PNG, or WEBP image."])
