"""Unit tests for ImagePreprocessor."""

import numpy as np
import pytest
from PIL import Image

from src.model.preprocessor import ImagePreprocessor, PreprocessingError


class TestImagePreprocessor:
    """Test suite for image preprocessing and validation."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor(target_size=(128, 128))

    def test_preprocess_jpeg_bytes(self, sample_image_bytes):
        tensor = self.preprocessor.decode_and_preprocess(sample_image_bytes)
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (1, 128, 128, 3)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_preprocess_png_bytes(self, sample_png_bytes):
        tensor = self.preprocessor.decode_and_preprocess(sample_png_bytes)
        assert tensor.shape == (1, 128, 128, 3)
        assert tensor.dtype == np.float32

    def test_preprocess_pil_image(self):
        pil_img = Image.new("RGB", (64, 64), color=(50, 100, 150))
        tensor = self.preprocessor.decode_and_preprocess(pil_img)
        assert tensor.shape == (1, 128, 128, 3)

    def test_preprocess_numpy_array(self):
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 200
        tensor = self.preprocessor.decode_and_preprocess(arr)
        assert tensor.shape == (1, 128, 128, 3)
        assert np.allclose(tensor.max(), 200 / 255.0, atol=1e-3)

    def test_empty_bytes_raises_error(self):
        with pytest.raises(PreprocessingError, match="cannot be empty"):
            self.preprocessor.decode_and_preprocess(b"")

    def test_corrupted_bytes_raises_error(self, corrupted_image_bytes):
        with pytest.raises(PreprocessingError):
            self.preprocessor.decode_and_preprocess(corrupted_image_bytes)

    def test_oversized_payload_raises_error(self):
        fake_large_payload = b"0" * (16 * 1024 * 1024)  # 16MB
        with pytest.raises(PreprocessingError, match="exceeds maximum"):
            self.preprocessor.validate_image_bytes(fake_large_payload, max_size_mb=15)
