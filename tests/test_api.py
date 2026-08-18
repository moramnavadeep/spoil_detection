"""Integration tests for FastAPI REST endpoints."""

import io
from fastapi import status


class TestAPIEndpoints:
    """Test suite for HTTP REST endpoints."""

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "online"
        assert "docs" in data

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "version" in data

    def test_model_metadata(self, client):
        response = client.get("/model/metadata")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["classes"] == ["Fresh", "Rotten"]
        assert data["input_shape"] == [128, 128, 3]

    def test_predict_single_image(self, client, sample_image_bytes):
        files = {"file": ("test_apple.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/predict", files=files)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["prediction"] in ["Fresh", "Rotten"]
        assert isinstance(data["is_fresh"], bool)
        assert data["filename"] == "test_apple.jpg"
        assert "probabilities" in data
        assert data["latency_ms"] >= 0

    def test_predict_batch_images(self, client, sample_image_bytes, sample_png_bytes):
        files = [
            ("files", ("img1.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")),
            ("files", ("img2.png", io.BytesIO(sample_png_bytes), "image/png")),
        ]
        response = client.post("/predict/batch", files=files)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_items"] == 2
        assert data["successful_items"] == 2
        assert data["failed_items"] == 0
        assert len(data["predictions"]) == 2

    def test_predict_invalid_content_type(self, client):
        files = {"file": ("document.txt", io.BytesIO(b"Hello world"), "text/plain")}
        response = client.post("/predict", files=files)
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_predict_corrupted_image(self, client, corrupted_image_bytes):
        files = {"file": ("corrupted.jpg", io.BytesIO(corrupted_image_bytes), "image/jpeg")}
        response = client.post("/predict", files=files)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
