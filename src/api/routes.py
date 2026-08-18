"""API Endpoints and Route Handlers."""

from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.api.schemas import (
    BatchItemResult,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadataResponse,
    SinglePredictionResult,
)
from src.config import settings
from src.model.inference import SpoilDetectionEngine
from src.model.preprocessor import PreprocessingError
from src.utils.logger import logger

router = APIRouter()


def get_engine() -> SpoilDetectionEngine:
    """Dependency provider for inference engine."""
    return SpoilDetectionEngine()


@router.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint to verify service uptime and model readiness."""
    engine = get_engine()
    model_loaded = engine.model is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Model Info"])
async def get_model_metadata():
    """Returns neural network architecture details, input dimensions, and target classes."""
    engine = get_engine()
    metadata = engine.get_metadata()
    return ModelMetadataResponse(**metadata)


@router.post(
    "/predict",
    response_model=SinglePredictionResult,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Classify a single food image for freshness",
)
async def predict_single_image(file: UploadFile = File(...)):
    """Accepts an image file (JPEG, PNG, WEBP) and returns spoilage prediction, confidence score, and latency."""
    # Validate content type if provided
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content-type '{file.content_type}'. Must be an image format.",
        )

    try:
        content = await file.read()
        engine = get_engine()
        result = engine.predict_image(content)
        result["filename"] = file.filename
        return SinglePredictionResult(**result)

    except PreprocessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image processing error: {str(exc)}",
        ) from exc
    except Exception as exc:
        logger.error("Inference endpoint failure: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Classify a batch of food images concurrently",
)
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """Accepts multiple image files and returns prediction results for each."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded.",
        )

    engine = get_engine()
    results: List[BatchItemResult] = []
    success_count = 0
    fail_count = 0

    for file in files:
        try:
            content = await file.read()
            res = engine.predict_image(content)
            res["filename"] = file.filename
            results.append(
                BatchItemResult(
                    filename=file.filename or "unknown",
                    success=True,
                    result=SinglePredictionResult(**res),
                    error=None,
                )
            )
            success_count += 1
        except Exception as exc:
            fail_count += 1
            results.append(
                BatchItemResult(
                    filename=file.filename or "unknown",
                    success=False,
                    result=None,
                    error=str(exc),
                )
            )

    return BatchPredictionResponse(
        total_items=len(files),
        successful_items=success_count,
        failed_items=fail_count,
        predictions=results,
    )
