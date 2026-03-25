"""Tests for PyTorchDataLoader.

Tests the NumPy-First architecture where:
- Datasets return raw NumPy arrays (no transforms)
- Transforms are applied in DataLoaders via FrameworkAdapter
- Albumentations transforms are used directly (no wrappers)
"""

import numpy as np
import pytest
import torch
from PIL import Image

from bovi_core.ml.dataloaders.datasets.image_dataset import ImageDataset
from bovi_core.ml.dataloaders.loaders.pytorch_loader import PyTorchDataLoader
from bovi_core.ml.dataloaders.sources.local_source import LocalFileSource

# Fixtures used from conftest:
# - image_dataset_large (from loaders/conftest.py)
# - mock_dataloader_config (from dataloaders/conftest.py)
# - albumentations_resize_transform (from loaders/conftest.py)


class TestPyTorchDataLoader:
    """Test PyTorchDataLoader with NumPy-First architecture."""

    def test_loader_initialization(self, image_dataset_large, mock_dataloader_config):
        """Test loader initialization."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=4,
            num_workers=0,
        )

        assert loader.batch_size == 4
        assert loader.split == "train"
        assert loader.model_name == "test_model"
        assert loader.num_workers >= 0
        assert loader._pytorch_loader is not None

    def test_loader_uses_config_defaults(self, image_dataset_large, mock_dataloader_config):
        """Test loader uses config defaults."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            num_workers=0,
        )

        # Should use config batch size
        assert loader.batch_size == 8
        assert loader.num_workers == 0  # Overridden by explicit param

    def test_loader_length(self, image_dataset_large, mock_dataloader_config):
        """Test loader returns correct number of batches."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=8,
            drop_last=False,
            num_workers=0,
        )

        # 40 images / batch_size=8 = 5 batches
        assert len(loader) == 5
        assert loader.num_batches == 5
        assert loader.num_samples == 40

    def test_loader_iteration_without_transform(self, image_dataset_large, mock_dataloader_config):
        """Test iterating over loader without transform."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=8,
            num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 5

        # Check batch structure
        batch = batches[0]
        assert "image" in batch
        assert "label" in batch

        # Without transform, images should be auto-converted
        # from (B, H, W, C) uint8 to (B, C, H, W) float32
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape[0] == 8  # Batch size
        assert batch["image"].shape[1] == 3  # Channels (auto-transposed)
        assert batch["image"].dtype == torch.float32  # Auto-normalized

    def test_loader_iteration_with_albumentations_transform(
        self, image_dataset_large, mock_dataloader_config, albumentations_resize_transform
    ):
        """Test iterating over loader WITH Albumentations transform."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            transform=albumentations_resize_transform,  # Transform passed to loader!
            batch_size=8,
            num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 5

        # Check batch structure
        batch = batches[0]
        assert "image" in batch
        assert "label" in batch

        # With transform, images should be resized and in PyTorch format
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape == (8, 3, 32, 32)  # (B, C, H, W)
        assert batch["image"].dtype == torch.float32

    def test_loader_shuffle(self, image_dataset_large, mock_dataloader_config):
        """Test shuffle parameter."""
        # Train should shuffle by default
        loader_train = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=8,
            num_workers=0,
        )
        assert loader_train.shuffle is True

        # Val should not shuffle by default
        loader_val = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="val",
            model_name="test_model",
            batch_size=8,
            num_workers=0,
        )
        assert loader_val.shuffle is False

        # Can override
        loader_custom = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="val",
            model_name="test_model",
            batch_size=8,
            shuffle=True,
            num_workers=0,
        )
        assert loader_custom.shuffle is True

    def test_loader_drop_last(self, image_dataset_large, mock_dataloader_config):
        """Test drop_last parameter."""
        # Without drop_last
        loader_keep = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=7,
            drop_last=False,
            num_workers=0,
        )
        # 40 images / 7 = 5 full batches + 1 partial (5 images)
        assert len(loader_keep) == 6

        # With drop_last
        loader_drop = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=7,
            drop_last=True,
            num_workers=0,
        )
        # Only 5 full batches
        assert len(loader_drop) == 5

    def test_loader_with_workers(self, image_dataset_large, mock_dataloader_config):
        """Test loader with multiple workers."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=8,
            num_workers=2,
        )

        assert loader.num_workers == 2

        # Should still iterate correctly
        batches = list(loader)
        assert len(batches) == 5

    def test_loader_pin_memory(self, image_dataset_large, mock_dataloader_config):
        """Test pin_memory auto-detection."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=8,
            num_workers=0,
        )

        # Should auto-detect based on CUDA availability
        assert isinstance(loader.pin_memory, bool)

        # Can override
        loader_pinned = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=8,
            pin_memory=True,
            num_workers=0,
        )
        assert loader_pinned.pin_memory is True

    def test_loader_persistent_workers(self, image_dataset_large, mock_dataloader_config):
        """Test persistent_workers parameter."""
        # Train with workers should have persistent workers
        loader_train = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=8,
            num_workers=2,
        )
        assert loader_train.persistent_workers is True

        # Val should not
        loader_val = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="val",
            model_name="test_model",
            batch_size=8,
            num_workers=2,
        )
        assert loader_val.persistent_workers is False

        # No workers should have persistent_workers=False
        loader_no_workers = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=8,
            num_workers=0,
        )
        assert loader_no_workers.persistent_workers is False

    def test_loader_get_pytorch_loader(self, image_dataset_large, mock_dataloader_config):
        """Test get_pytorch_loader returns DataLoader."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            num_workers=0,
        )

        pytorch_loader = loader.get_pytorch_loader()
        assert pytorch_loader is not None
        assert hasattr(pytorch_loader, "__iter__")

    def test_loader_get_tensorflow_dataset(self, image_dataset_large, mock_dataloader_config):
        """Test get_tensorflow_dataset returns None."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            num_workers=0,
        )

        assert loader.get_tensorflow_dataset() is None

    def test_loader_get_sklearn_iterator(self, image_dataset_large, mock_dataloader_config):
        """Test get_sklearn_iterator returns None."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            num_workers=0,
        )

        assert loader.get_sklearn_iterator() is None

    def test_loader_collate_function(self, image_dataset_large, mock_dataloader_config):
        """Test custom collate function handles various types."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=4,
            num_workers=0,
        )

        batch = next(iter(loader))

        # Images should be stacked tensors
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape[0] == 4  # Batch size

        # Labels should be list (strings can't be stacked)
        assert isinstance(batch["label"], list)
        assert len(batch["label"]) == 4

    def test_loader_multiple_epochs(self, image_dataset_large, mock_dataloader_config):
        """Test loader can iterate multiple epochs."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=8,
            num_workers=0,
        )

        # First epoch
        epoch1_batches = list(loader)
        assert len(epoch1_batches) == 5

        # Second epoch
        epoch2_batches = list(loader)
        assert len(epoch2_batches) == 5

        # Third epoch
        epoch3_batches = list(loader)
        assert len(epoch3_batches) == 5

    def test_loader_prefetch_factor(self, image_dataset_large, mock_dataloader_config):
        """Test prefetch_factor parameter."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=8,
            num_workers=2,
            prefetch_factor=4,
        )

        assert loader.prefetch_factor == 4

        # Should still work
        batches = list(loader)
        assert len(batches) == 5

    def test_loader_empty_dataset(self, mock_dataloader_config, tmp_path):
        """Test loader with empty dataset."""
        # Create empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        source = LocalFileSource(empty_dir, file_pattern="*.jpg")
        dataset = ImageDataset(source)

        # Empty dataset with shuffle=True will fail in PyTorch (RandomSampler)
        # So we explicitly use shuffle=False
        loader = PyTorchDataLoader(
            dataset,
            config=mock_dataloader_config,
            split="val",
            model_name="test_model",
            batch_size=8,
            num_workers=0,
            shuffle=False,  # Explicit: PyTorch RandomSampler fails on empty dataset
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

        loader = PyTorchDataLoader(
            dataset,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=4,
            num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 1
        assert batches[0]["image"].shape[0] == 1  # Batch size of 1

    def test_loader_auto_transpose_disabled(self, image_dataset_large, mock_dataloader_config):
        """Test disabling auto-transpose keeps HWC format."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=4,
            num_workers=0,
            auto_transpose=False,
        )

        batch = next(iter(loader))

        # Without auto_transpose, images stay in (B, H, W, C) format
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape[0] == 4  # Batch size
        assert batch["image"].shape[-1] == 3  # Channels last (HWC)

    def test_loader_auto_normalize_disabled(self, image_dataset_large, mock_dataloader_config):
        """Test disabling auto-normalize keeps uint8."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=4,
            num_workers=0,
            auto_normalize=False,
        )

        batch = next(iter(loader))

        # Without auto_normalize, images stay as uint8
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].dtype == torch.uint8

    def test_loader_transform_parameter(
        self, image_dataset_large, mock_dataloader_config, albumentations_resize_transform
    ):
        """Test that transform is stored as loader attribute."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            transform=albumentations_resize_transform,
            batch_size=4,
            num_workers=0,
        )

        # Transform is stored on the loader, not the dataset
        assert loader.transform is albumentations_resize_transform
