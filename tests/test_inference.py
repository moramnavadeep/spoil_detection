"""Unit tests for SpoilDetectionEngine."""

import pytest
from src.model.inference import SpoilDetectionEngine


class TestSpoilDetectionEngine:
    """Test suite for deep learning inference engine."""

    def test_singleton_instance(self, inference_engine):
        engine2 = SpoilDetectionEngine()
        assert inference_engine is engine2

    def test_metadata_structure(self, inference_engine):
        meta = inference_engine.get_metadata()
        assert "model_name" in meta
        assert "version" in meta
        assert meta["classes"] == ["Fresh", "Rotten"]
        assert meta["input_shape"] == [128, 128, 3]
        assert meta["total_parameters"] > 0

    def test_predict_single_image(self, inference_engine, sample_image_bytes):
        result = inference_engine.predict_image(sample_image_bytes)

        assert "prediction" in result
        assert result["prediction"] in ["Fresh", "Rotten"]
        assert isinstance(result["is_fresh"], bool)
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["confidence_percentage"] <= 100.0
        assert "probabilities" in result
        assert "Fresh" in result["probabilities"]
        assert "Rotten" in result["probabilities"]
        prob_sum = sum(result["probabilities"].values())
        assert pytest.approx(prob_sum, rel=1e-2) == 1.0
        assert result["latency_ms"] > 0

    def test_predict_batch_images(self, inference_engine, sample_image_bytes, sample_png_bytes):
        batch = [sample_image_bytes, sample_png_bytes]
        results = inference_engine.predict_batch(batch)

        assert len(results) == 2
        for item in results:
            assert item["success"] is True
            assert item["result"]["prediction"] in ["Fresh", "Rotten"]

    def test_predict_corrupted_in_batch(self, inference_engine, sample_image_bytes, corrupted_image_bytes):
        batch = [sample_image_bytes, corrupted_image_bytes]
        results = inference_engine.predict_batch(batch)

        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[1]["error"] is not None
