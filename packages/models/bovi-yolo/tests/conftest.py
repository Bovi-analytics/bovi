"""Shared test fixtures for YOLO model tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

if TYPE_CHECKING:
    from bovi_core.config import Config


@pytest.fixture
def yolo_config() -> Config:
    """Create a Config instance for the YOLO experiment.

    Uses Config.reset() to ensure a clean singleton for each test.

    Returns:
        Config instance loaded from data/experiments/yolo/versions/v1/config/config.yaml.
    """
    from bovi_core.config import Config

    Config.reset()
    return Config(experiment_name="yolo", project_name="bovi-yolo")


@pytest.fixture
def temp_image_dir(tmp_path: Path) -> Path:
    """Create temp directory with synthetic test images (train/val/test splits).

    Returns:
        Path to root directory containing split subdirectories.
    """
    for split in ("train", "val", "test"):
        split_dir = tmp_path / split / "images"
        split_dir.mkdir(parents=True)

        # Create a synthetic image per split
        img = Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        suffix = ".jpeg" if split == "train" else ".jpg"
        img.save(split_dir / f"cow_{split}{suffix}")

    return tmp_path


@pytest.fixture
def sample_image() -> npt.NDArray[np.uint8]:
    """Create a sample RGB image array."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_small() -> npt.NDArray[np.uint8]:
    """Create a small sample RGB image array."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_boxes() -> npt.NDArray[np.float64]:
    """Create sample bounding boxes in xyxy format."""
    return np.array(
        [
            [100.0, 50.0, 300.0, 250.0],
            [400.0, 100.0, 550.0, 300.0],
        ]
    )


@pytest.fixture
def sample_scores() -> npt.NDArray[np.float64]:
    """Create sample confidence scores."""
    return np.array([0.95, 0.78])


@pytest.fixture
def sample_class_ids() -> npt.NDArray[np.float64]:
    """Create sample class IDs."""
    return np.array([0.0, 0.0])


@pytest.fixture
def sample_class_names() -> list[str]:
    """Create sample class names."""
    return ["cow", "cow"]
