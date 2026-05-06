"""Tests for TensorFlowDataLoader.

Tests the NumPy-First architecture where:
- Datasets return raw NumPy arrays (no transforms)
- Transforms are applied in DataLoaders
- Albumentations transforms are used directly (no wrappers)
"""

import numpy as np

# Fixtures used from conftest:
# - image_dataset_large (from loaders/conftest.py)
# - mock_dataloader_config (from dataloaders/conftest.py)
# - albumentations_resize_transform (from loaders/conftest.py)
import tensorflow as tf
from bovi_core.ml.dataloaders.datasets.image_dataset import ImageDataset
from bovi_core.ml.dataloaders.loaders.tensorflow_loader import TensorFlowDataLoader
from bovi_core.ml.dataloaders.sources.local_source import LocalFileSource
from PIL import Image


class TestTensorFlowDataLoader:
    """Test TensorFlowDataLoader with NumPy-First architecture."""

    def test_loader_initialization(self, image_dataset_large, mock_dataloader_config):
        """Test loader initialization."""
        loader = TensorFlowDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=4,
        )

        assert loader.batch_size == 4
        assert loader.split == "train"
        assert loader._tf_dataset is not None

    def test_loader_uses_config_defaults(self, image_dataset_large, mock_dataloader_config):
        """Test loader uses config defaults."""
        loader = TensorFlowDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
        )

        # Should use config batch size
        assert loader.batch_size == 8

    def test_loader_length(self, image_dataset_large, mock_dataloader_config):
        """Test loader returns correct number of batches."""
        loader = TensorFlowDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train", batch_size=8
        )

        # 40 images / batch_size=8 = 5 batches
        assert len(loader) == 5
        assert loader.num_batches == 5
        assert loader.num_samples == 40

    def test_loader_iteration_without_transform(self, image_dataset_large, mock_dataloader_config):
        """Test iterating over loader without transform."""

        loader = TensorFlowDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train", batch_size=8
        )

        batches = list(loader)
        assert len(batches) == 5

        # Check batch structure
        batch = batches[0]
        assert "image" in batch
        assert "label" in batch

        # Check batch shapes - TensorFlow keeps BHWC format
        assert batch["image"].shape == (8, 64, 64, 3)  # (B, H, W, C) for TF
        assert len(batch["label"]) == 8

    def test_loader_iteration_with_albumentations_transform(
        self, image_dataset_large, mock_dataloader_config, albumentations_resize_transform
    ):
        """Test iterating over loader WITH Albumentations transform."""

        loader = TensorFlowDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            transform=albumentations_resize_transform,  # Transform passed to loader!
            batch_size=8,
        )

        batches = list(loader)
        assert len(batches) == 5

        # Check batch structure
        batch = batches[0]
        assert "image" in batch
        assert "label" in batch

        # With transform, images should be resized
        assert batch["image"].shape == (8, 32, 32, 3)  # (B, H, W, C) for TF
        # TensorFlow loader normalizes uint8 to float32
        assert batch["image"].dtype == tf.float32

    def test_loader_shuffle(self, image_dataset_large, mock_dataloader_config):
        """Test shuffle parameter."""
        # Train should shuffle by default
        loader_train = TensorFlowDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train", batch_size=8
        )
        assert loader_train.shuffle is True

        # Val should not shuffle by default
        loader_val = TensorFlowDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="val", batch_size=8
        )
        assert loader_val.shuffle is False

    def test_loader_get_tensorflow_dataset(self, image_dataset_large, mock_dataloader_config):
        """Test get_tensorflow_dataset returns Dataset."""

        loader = TensorFlowDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train"
        )

        tf_dataset = loader.get_tensorflow_dataset()
        assert tf_dataset is not None
        assert isinstance(tf_dataset, tf.data.Dataset)

    def test_loader_get_pytorch_loader(self, image_dataset_large, mock_dataloader_config):
        """Test get_pytorch_loader returns None."""
        loader = TensorFlowDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train"
        )

        assert loader.get_pytorch_loader() is None

    def test_loader_get_sklearn_iterator(self, image_dataset_large, mock_dataloader_config):
        """Test get_sklearn_iterator returns None."""
        loader = TensorFlowDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train"
        )

        assert loader.get_sklearn_iterator() is None

    def test_loader_multiple_epochs(self, image_dataset_large, mock_dataloader_config):
        """Test loader can iterate multiple epochs."""
        loader = TensorFlowDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train", batch_size=8
        )

        # First epoch
        epoch1_batches = list(loader)
        assert len(epoch1_batches) == 5

        # Second epoch
        epoch2_batches = list(loader)
        assert len(epoch2_batches) == 5

    def test_loader_cache(self, image_dataset_large, mock_dataloader_config):
        """Test caching parameter."""
        loader = TensorFlowDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            batch_size=8,
            cache=True,
        )

        assert loader.cache is True

        # Should still work
        batches = list(loader)
        assert len(batches) == 5

    def test_loader_transform_parameter(
        self, image_dataset_large, mock_dataloader_config, albumentations_resize_transform
    ):
        """Test that transform is stored as loader attribute."""
        loader = TensorFlowDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            transform=albumentations_resize_transform,
            batch_size=4,
        )

        # Transform is stored on the loader, not the dataset
        assert loader.transform is albumentations_resize_transform

    def test_loader_element_spec(self, image_dataset_large, mock_dataloader_config):
        """Test element_spec property for shape debugging."""
        loader = TensorFlowDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train", batch_size=8
        )

        # element_spec should be available for shape inspection
        spec = loader.element_spec
        assert spec is not None
        assert "image" in spec
        assert "label" in spec

    def test_loader_output_shapes_inferred(
        self, image_dataset_large, mock_dataloader_config, albumentations_resize_transform
    ):
        """Test that output shapes are correctly inferred (dry run)."""
        loader = TensorFlowDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            transform=albumentations_resize_transform,
            batch_size=4,
        )

        # Check that shapes were inferred correctly
        assert "image" in loader._output_shapes
        assert loader._output_shapes["image"] == (32, 32, 3)

    def test_loader_empty_dataset(self, mock_dataloader_config, tmp_path):
        """Test loader with empty dataset."""
        # Create empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        source = LocalFileSource(empty_dir, file_pattern="*.jpg")
        dataset = ImageDataset(source)

        loader = TensorFlowDataLoader(
            dataset,
            config=mock_dataloader_config,
            split="val",  # val defaults to shuffle=False
            batch_size=8,
        )

        assert len(loader) == 0
        batches = list(loader)
        assert len(batches) == 0

    def test_loader_single_sample(self, mock_dataloader_config, tmp_path):
        """Test loader with single sample."""
        # Create single image
        single_dir = tmp_path / "single" / "class"
        single_dir.mkdir(parents=True)

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(single_dir / "img.jpg")

        source = LocalFileSource(tmp_path / "single", file_pattern="*.jpg")
        # NumPy-First: Dataset has no transforms
        dataset = ImageDataset(source)

        loader = TensorFlowDataLoader(
            dataset,
            config=mock_dataloader_config,
            split="train",
            batch_size=4,
        )

        batches = list(loader)
        assert len(batches) == 1
        assert batches[0]["image"].shape[0] == 1  # Batch size of 1
