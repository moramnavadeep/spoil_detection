"""Unit tests for CLI commands."""

import subprocess
import sys
from pathlib import Path


def test_cli_help():
    """Verifies that the CLI help message renders properly."""
    result = subprocess.run(
        [sys.executable, "-m", "src.cli.main", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "NutriFresh AI" in result.stdout
    assert "predict" in result.stdout
    assert "evaluate" in result.stdout
    assert "serve" in result.stdout


def test_cli_predict_sample(sample_image_bytes, tmp_path):
    """Verifies CLI single image prediction on temporary image."""
    test_img = tmp_path / "test.jpg"
    test_img.write_bytes(sample_image_bytes)

    result = subprocess.run(
        [sys.executable, "-m", "src.cli.main", "predict", "--image", str(test_img), "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert '"prediction"' in result.stdout
    assert '"confidence"' in result.stdout
