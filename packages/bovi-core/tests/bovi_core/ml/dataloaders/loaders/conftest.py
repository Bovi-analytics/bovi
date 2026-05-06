"""Fixtures specific to loader tests."""

import pytest
from bovi_core.ml.dataloaders.datasets.image_dataset import ImageDataset


@pytest.fixture
def image_dataset_large(image_source_large):
    """ImageDataset with large source (40 images) for batch testing."""
    return ImageDataset(image_source_large)


@pytest.fixture
def albumentations_resize_transform():
    """Albumentations transform that resizes to 32x32."""
    try:
        import albumentations as A

        return A.Compose([A.Resize(32, 32)])
    except ImportError:
        pytest.skip("Albumentations not installed")
