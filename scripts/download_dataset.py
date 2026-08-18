"""Automated Food Freshness Dataset Downloader.

Downloads the full 4.1GB food freshness dataset from KaggleHub into the local datasets directory.

Usage:
    python scripts/download_dataset.py
"""

import os
import sys
from pathlib import Path


def download_dataset():
    """Downloads the Kaggle dataset using kagglehub."""
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Installing it now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub

    print("Downloading 'ulnnproject/food-freshness-dataset' from KaggleHub...")
    dataset_path = kagglehub.dataset_download("ulnnproject/food-freshness-dataset")
    print(f"\n✅ Dataset successfully downloaded to: {dataset_path}")
    print("\nTo train the model on this dataset, run:")
    print(f'python train.py --dataset-path "{dataset_path}" --epochs 15')


if __name__ == "__main__":
    download_dataset()
