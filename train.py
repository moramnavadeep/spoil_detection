"""Standalone Training and Evaluation Pipeline for Food Spoilage Classification.

Usage:
    python train.py --dataset-path ./datasets/food-freshness --epochs 15 --batch-size 32 --output-model spoil_detection_model.h5
"""

import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import settings
from src.model.architecture import build_spoil_detection_model
from src.utils.logger import logger


def train_model(
    dataset_path: str,
    epochs: int = 15,
    batch_size: int = 32,
    img_size: int = 128,
    output_model_path: str = "spoil_detection_model.h5",
    plots_dir: str = "docs/plots",
):
    """Trains the CNN model on the specified food freshness dataset."""
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

    logger.info("Setting up ImageDataGenerators with augmentation...")
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        str(dataset_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = train_datagen.flow_from_directory(
        str(dataset_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    num_classes = len(train_generator.class_indices)
    logger.info("Found %d classes: %s", num_classes, list(train_generator.class_indices.keys()))

    # Build CNN Architecture
    model = build_spoil_detection_model(
        input_shape=(img_size, img_size, 3),
        num_classes=num_classes,
        dropout_rate=0.5,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=output_model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    logger.info("Starting training for %d epochs...", epochs)
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
    )

    # Save final model
    model.save(output_model_path)
    logger.info("Training complete. Best model saved to %s", output_model_path)

    # Export loss and accuracy curves
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_file = os.path.join(plots_dir, "training_metrics.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    logger.info("Training metrics curve saved to %s", plot_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spoil Detection CNN Model")
    parser.add_argument("--dataset-path", default="datasets/ulnnproject/food-freshness-dataset/versions/1/Dataset", help="Path to dataset root folder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--img-size", type=int, default=128, help="Target image dimension (square)")
    parser.add_argument("--output-model", default="spoil_detection_model.h5", help="Output path for trained weights")

    args = parser.parse_args()
    train_model(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        output_model_path=args.output_model,
    )
