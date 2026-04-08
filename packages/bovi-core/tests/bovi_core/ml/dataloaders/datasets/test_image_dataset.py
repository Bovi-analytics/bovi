"""Tests for ImageDataset.

Tests the NumPy-First architecture where:
- Datasets return raw NumPy arrays (no transforms)
- Transforms are applied in DataLoaders via FrameworkAdapter
"""

import numpy as np
from bovi_core.ml.dataloaders.datasets.image_dataset import ImageDataset
from bovi_core.ml.dataloaders.sources.local_source import LocalFileSource
from PIL import Image


class TestImageDataset:
    """Test ImageDataset with NumPy-First architecture."""

    def test_dataset_length(self, image_dataset):
        """Test dataset returns correct length."""
        assert len(image_dataset) == 3

    def test_dataset_getitem_returns_numpy(self, image_dataset):
        """Test that getitem returns NumPy arrays (not PIL Images)."""
        item = image_dataset[0]

        assert "image" in item
        assert "label" in item
        # NumPy-First: Always returns NumPy array, not PIL Image
        assert isinstance(item["image"], np.ndarray)
        assert item["image"].dtype == np.uint8
        assert item["image"].shape == (64, 64, 3)  # HWC format

    def test_dataset_getitem_with_metadata(self, image_source):
        """Test getitem with metadata."""
        dataset = ImageDataset(image_source, return_metadata=True)
        item = dataset[0]

        assert "image" in item
        assert "label" in item
        assert "metadata" in item
        assert "path" in item["metadata"]
        assert "size_bytes" in item["metadata"]

    def test_dataset_label_counts(self, image_dataset):
        """Test label counting."""
        label_counts = image_dataset.get_label_counts()

        assert "cat" in label_counts
        assert "dog" in label_counts
        assert label_counts["cat"] == 2
        assert label_counts["dog"] == 1

    def test_dataset_get_sample(self, image_dataset):
        """Test get_sample method."""
        sample = image_dataset.get_sample(0)

        assert "image" in sample
        assert "label" in sample
        assert isinstance(sample["image"], np.ndarray)

    def test_dataset_iteration(self, image_dataset):
        """Test iterating over dataset."""
        items = list(image_dataset)

        assert len(items) == 3
        for item in items:
            assert "image" in item
            assert "label" in item
            assert isinstance(item["image"], np.ndarray)

    def test_dataset_rgba_conversion(self, tmp_path):
        """Test that RGBA images are converted to RGB."""
        # Create RGBA image
        rgba_dir = tmp_path / "rgba"
        rgba_dir.mkdir()

        rgba_img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8), mode="RGBA"
        )
        rgba_img.save(rgba_dir / "test.png")

        source = LocalFileSource(rgba_dir, file_pattern="*.png")
        dataset = ImageDataset(source)
        item = dataset[0]

        # Should be converted to RGB (3 channels)
        assert isinstance(item["image"], np.ndarray)
        assert item["image"].shape == (64, 64, 3)

    def test_dataset_grayscale_conversion(self, tmp_path):
        """Test that grayscale images are converted to RGB."""
        # Create grayscale image
        gray_dir = tmp_path / "gray"
        gray_dir.mkdir()

        gray_img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L")
        gray_img.save(gray_dir / "test.jpg")

        source = LocalFileSource(gray_dir, file_pattern="*.jpg")
        dataset = ImageDataset(source)
        item = dataset[0]

        # Should be converted to RGB (3 channels)
        assert isinstance(item["image"], np.ndarray)
        assert item["image"].shape == (64, 64, 3)

    def test_dataset_without_labels(self, tmp_path):
        """Test dataset when labels are not available."""
        # Create flat directory (no label structure)
        flat_dir = tmp_path / "images"
        flat_dir.mkdir()

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(flat_dir / "test.jpg")

        source = LocalFileSource(flat_dir, file_pattern="*.jpg")
        dataset = ImageDataset(source)
        item = dataset[0]

        assert isinstance(item["image"], np.ndarray)
        # Label will be derived from directory name
        assert item["label"] is not None

    def test_dataset_no_transform_argument(self, image_dataset):
        """Test that Dataset does not accept transform argument (NumPy-First)."""
        # ImageDataset should NOT have a transform parameter
        # The signature is: __init__(source, config=None, return_metadata=False)
        assert (
            not hasattr(image_dataset, "transform")
            or image_dataset.__dict__.get("transform") is None
        )

    def test_dataset_returns_uint8(self, image_dataset):
        """Test that dataset returns uint8 images (no normalization)."""
        item = image_dataset[0]

        # Raw NumPy, no normalization
        assert item["image"].dtype == np.uint8
        assert item["image"].min() >= 0
        assert item["image"].max() <= 255
