"""Fixtures specific to loader tests."""

import numpy as np
import pytest
from bovi_core.ml.dataloaders.datasets.image_dataset import ImageDataset


@pytest.fixture
def dense_samples():
    """Framework-neutral nested records; existing image fixtures cannot cover these."""
    return [
        {
            "features": {
                "nested": {"vector": np.array([i, i + 1], dtype=np.float64)},
                "sequence": [i, i + 2],
                "pixels": np.full((2, 2, 3), i, dtype=np.uint8),
                "enabled": np.bool_(i % 2),
            },
            "labels": np.float32(i + 0.5),
            "metadata": {"id": str(i), "nested": {"index": i}},
        }
        for i in range(3)
    ]


@pytest.fixture
def shuffle_samples():
    """Distinct records so shuffle tests detect reordered samples, not just labels."""
    return [{"features": np.int64(index)} for index in range(32)]


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
