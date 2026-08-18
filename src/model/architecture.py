"""Convolutional Neural Network Architecture for Food Spoilage Classification."""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


def build_spoil_detection_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    """Constructs the sequential CNN architecture used for food spoilage detection.

    Architecture Hierarchy:
        - Block 1: Conv2D(32, 3x3, ReLU) -> MaxPooling2D(2x2)
        - Block 2: Conv2D(64, 3x3, ReLU) -> MaxPooling2D(2x2)
        - Block 3: Conv2D(128, 3x3, ReLU) -> MaxPooling2D(2x2)
        - Dense Head: Flatten -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(num_classes, Softmax)

    Args:
        input_shape: Shape of input image tensor (H, W, C).
        num_classes: Number of target categories (e.g. 2 for Fresh vs Rotten).
        dropout_rate: Dropout fraction for dense regularization.

    Returns:
        Compiled or uncompiled tf.keras.Model.
    """
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape, name="conv_block1"),
            MaxPooling2D((2, 2), name="pool_block1"),
            Conv2D(64, (3, 3), activation="relu", name="conv_block2"),
            MaxPooling2D((2, 2), name="pool_block2"),
            Conv2D(128, (3, 3), activation="relu", name="conv_block3"),
            MaxPooling2D((2, 2), name="pool_block3"),
            Flatten(name="flatten"),
            Dense(128, activation="relu", name="dense_features"),
            Dropout(dropout_rate, name="dropout_reg"),
            Dense(num_classes, activation="softmax", name="output_probabilities"),
        ],
        name="NutriFresh_CNN",
    )

    return model
