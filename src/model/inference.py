"""Deep Learning Inference Engine for Spoilage Detection."""

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import tensorflow as tf

from src.config import settings
from src.model.preprocessor import ImagePreprocessor, PreprocessingError
from src.utils.logger import logger


class ModelInferenceError(Exception):
    """Custom exception raised during model inference."""
    pass


class SpoilDetectionEngine:
    """Singleton inference engine managing model lifecycle, thread safety, and inference."""

    _instance: Optional["SpoilDetectionEngine"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SpoilDetectionEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        if getattr(self, "_initialized", False):
            return

        self.model_path = Path(model_path or settings.MODEL_PATH)
        self.classes = settings.CLASSES
        self.preprocessor = ImagePreprocessor()
        self.model: Optional[tf.keras.Model] = None
        self._infer_lock = threading.Lock()

        self._load_model()
        self._warmup()
        self._initialized = True

    def _load_model(self) -> None:
        """Loads the compiled Keras / HDF5 model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        logger.info("Loading Spoil Detection model from: %s", self.model_path)
        try:
            # Disable unnecessary oneDNN / TF verbose logs
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            logger.info("Model loaded successfully into memory.")
        except Exception as exc:
            logger.error("Failed to load model: %s", str(exc), exc_info=True)
            raise ModelInferenceError(f"Failed to load model weights from {self.model_path}: {exc}") from exc

    def _warmup(self) -> None:
        """Runs a synthetic warm-up forward pass to initialize graph execution and GPU/CPU caches."""
        if self.model is None:
            return

        try:
            logger.info("Running inference warmup pass...")
            dummy_input = np.zeros((1, settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            logger.info("Model warmup complete.")
        except Exception as exc:
            logger.warning("Warmup pass failed (non-critical): %s", str(exc))

    def predict_image(self, image_input: Union[bytes, np.ndarray, Any]) -> Dict[str, Any]:
        """Runs end-to-end inference on a single image.

        Args:
            image_input: Raw image bytes, PIL Image, or NumPy array.

        Returns:
            Dictionary containing prediction class, confidence, probabilities, and latency.

        Raises:
            PreprocessingError: If image validation or preprocessing fails.
            ModelInferenceError: If model prediction fails.
        """
        if self.model is None:
            raise ModelInferenceError("Model is not initialized.")

        # 1. Preprocess
        tensor = self.preprocessor.decode_and_preprocess(image_input)

        # 2. Predict with latency timing
        start_time = time.perf_counter()
        with self._infer_lock:
            raw_preds = self.model.predict(tensor, verbose=0)[0]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Format outputs
        class_idx = int(np.argmax(raw_preds))
        predicted_label = self.classes[class_idx]
        confidence = float(raw_preds[class_idx])

        probabilities = {
            self.classes[i]: float(raw_preds[i]) for i in range(len(self.classes))
        }

        is_fresh = predicted_label.lower() == "fresh"

        return {
            "prediction": predicted_label,
            "is_fresh": is_fresh,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100.0, 2),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "latency_ms": round(latency_ms, 2),
            "model_version": settings.VERSION,
        }

    def predict_batch(self, batch_inputs: List[Union[bytes, np.ndarray, Any]]) -> List[Dict[str, Any]]:
        """Processes a batch of images and returns prediction results for each."""
        results = []
        for item in batch_inputs:
            try:
                res = self.predict_image(item)
                results.append({"success": True, "result": res, "error": None})
            except Exception as exc:
                results.append({"success": False, "result": None, "error": str(exc)})
        return results

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the loaded model and its runtime."""
        total_params = int(self.model.count_params()) if self.model else 0
        return {
            "model_name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "architecture": "Convolutional Neural Network (CNN - 3 Blocks)",
            "input_shape": [settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS],
            "classes": self.classes,
            "total_parameters": total_params,
            "model_format": self.model_path.suffix,
        }
