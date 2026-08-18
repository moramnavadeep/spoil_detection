"""Pytest fixtures and configuration."""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.model.inference import SpoilDetectionEngine


@pytest.fixture(scope="session")
def app():
    """FastAPI application fixture."""
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def inference_engine():
    """Singleton inference engine fixture."""
    return SpoilDetectionEngine()


@pytest.fixture
def sample_image_bytes():
    """Generates synthetic JPEG image bytes for fast unit testing."""
    img = Image.new("RGB", (200, 200), color=(120, 200, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes():
    """Generates synthetic PNG image bytes."""
    img = Image.new("RGB", (150, 150), color=(220, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def corrupted_image_bytes():
    """Generates invalid bytes representing corrupted data."""
    return b"not-a-valid-image-binary-stream-corrupted"
