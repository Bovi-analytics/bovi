"""Shared fixtures for all dataloader tests."""

from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from bovi_core.ml.dataloaders.sources.local_source import LocalFileSource


# --- Config Mock (used by all loaders) ---


@pytest.fixture
def mock_dataloader_config():
    """Mock config with experiment.models structure for dataloader tests."""
    config = Mock()
    config.experiment = Mock()
    config.experiment.models = Mock()

    # Mock a test model with dataloaders
    test_model = Mock()
    test_model.dataloaders = Mock()

    # Train split config with nested dataloader
    train_config = Mock()
    train_config.dataloader = Mock()
    train_config.dataloader.batch_size = 8
    train_config.dataloader.num_workers = 2
    train_config.dataloader.shuffle = True
    train_config.configure_mock(**{"dataloader": train_config.dataloader})
    test_model.dataloaders.train = train_config

    # Val split config (shuffle=False by default for validation)
    val_config = Mock()
    val_config.dataloader = Mock()
    val_config.dataloader.batch_size = 8
    val_config.dataloader.num_workers = 2
    val_config.dataloader.shuffle = False
    val_config.configure_mock(**{"dataloader": val_config.dataloader})
    test_model.dataloaders.val = val_config

    config.experiment.models.test_model = test_model
    return config


# --- Image Test Data ---


@pytest.fixture
def temp_image_dir(tmp_path):
    """Create small temp directory with test images (3 images).

    Structure:
        tmp/
        ├── cat/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── dog/
            └── img3.jpg
    """
    cat_dir = tmp_path / "cat"
    dog_dir = tmp_path / "dog"
    cat_dir.mkdir()
    dog_dir.mkdir()

    # Create 64x64 RGB images
    for i, dir_path in enumerate([cat_dir, cat_dir, dog_dir]):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(dir_path / f"img{i+1}.jpg")

    return tmp_path


@pytest.fixture
def temp_image_dir_large(tmp_path):
    """Create large temp directory with test images (40 images, 20 per class).

    Suitable for loader batch tests with batch_size=8 (5 batches).
    """
    cat_dir = tmp_path / "cat"
    dog_dir = tmp_path / "dog"
    cat_dir.mkdir()
    dog_dir.mkdir()

    # Create 64x64 RGB images
    for i in range(20):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(cat_dir / f"cat_{i}.jpg")

    for i in range(20):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(dog_dir / f"dog_{i}.jpg")

    return tmp_path


# --- Sources ---


@pytest.fixture
def image_source(temp_image_dir):
    """LocalFileSource with small image dataset (3 images)."""
    return LocalFileSource(temp_image_dir, file_pattern="*.jpg")


@pytest.fixture
def image_source_large(temp_image_dir_large):
    """LocalFileSource with large image dataset (40 images)."""
    return LocalFileSource(temp_image_dir_large, file_pattern="*.jpg")
