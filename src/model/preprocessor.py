"""Image Preprocessing and Validation Pipeline."""

import io
from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.config import settings
from src.utils.logger import logger


class PreprocessingError(Exception):
    """Custom exception raised during image preprocessing failures."""
    pass


class ImagePreprocessor:
    """Handles image validation, decoding, resizing, and normalization for the model."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (settings.IMG_WIDTH, settings.IMG_HEIGHT),
        channels: int = settings.IMG_CHANNELS,
    ):
        self.target_size = target_size
        self.channels = channels

    def validate_image_bytes(self, image_bytes: bytes, max_size_mb: int = settings.MAX_UPLOAD_SIZE_MB) -> None:
        """Validates payload size and non-emptiness.

        Args:
            image_bytes: Raw binary bytes of the image.
            max_size_mb: Maximum allowed image size in megabytes.

        Raises:
            PreprocessingError: If input is empty or exceeds size limit.
        """
        if not image_bytes:
            raise PreprocessingError("Image payload cannot be empty.")

        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise PreprocessingError(
                f"Image size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)."
            )

    def decode_and_preprocess(self, image_input: Union[bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """Decodes raw input, resizes to target dimension, and normalizes pixel values to [0, 1].

        Args:
            image_input: Raw image bytes, PIL Image, or NumPy array.

        Returns:
            Preprocessed 4D NumPy batch array of shape (1, H, W, C) with float32 values in [0, 1].

        Raises:
            PreprocessingError: If the image cannot be decoded or processed.
        """
        try:
            if isinstance(image_input, bytes):
                self.validate_image_bytes(image_input)
                # Decode bytes to OpenCV BGR image
                np_arr = np.frombuffer(image_input, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    # Fallback to PIL in case cv2 failed on webp/special formats
                    pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
                    img_rgb = np.array(pil_img)
                    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            elif isinstance(image_input, Image.Image):
                img_rgb = np.array(image_input.convert("RGB"))
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            elif isinstance(image_input, np.ndarray):
                img = image_input
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            else:
                raise PreprocessingError(f"Unsupported image input type: {type(image_input)}")

            if img is None or img.size == 0:
                raise PreprocessingError("Failed to decode image. Corrupted or invalid format.")

            # Resize to expected model dimensions (width, height)
            resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

            # Convert to float32 and normalize [0, 1]
            normalized = resized.astype(np.float32) / 255.0

            # Expand dimensions to create batch shape (1, H, W, C)
            batch_tensor = np.expand_dims(normalized, axis=0)

            return batch_tensor

        except PreprocessingError:
            raise
        except Exception as exc:
            logger.error("Preprocessing error: %s", str(exc), exc_info=True)
            raise PreprocessingError(f"Failed to process image: {str(exc)}") from exc
