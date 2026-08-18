"""Application Configuration Module."""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    """Application settings and configuration parameters."""

    # Project metadata
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "NutriFresh AI - Food Spoilage Detection")
    VERSION: str = os.getenv("VERSION", "1.0.0")
    DESCRIPTION: str = os.getenv(
        "DESCRIPTION",
        "Deep Learning Computer Vision API for automated food freshness classification and quality assessment.",
    )
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    # Model Configuration
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "spoil_detection_model.h5")))
    IMG_HEIGHT: int = int(os.getenv("IMG_HEIGHT", "128"))
    IMG_WIDTH: int = int(os.getenv("IMG_WIDTH", "128"))
    IMG_CHANNELS: int = int(os.getenv("IMG_CHANNELS", "3"))
    CLASSES: List[str] = ["Fresh", "Rotten"]

    # Inference & Thresholds
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15"))

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS: List[str] = [
        origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()
    ]

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Path = BASE_DIR / "logs" / "app.log"


settings = Settings()
