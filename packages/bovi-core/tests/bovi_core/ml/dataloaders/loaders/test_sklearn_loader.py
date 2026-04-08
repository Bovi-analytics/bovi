"""Tests for SklearnDataLoader.

Tests the NumPy-First architecture where:
- Datasets return raw NumPy arrays
- SklearnDataLoader provides simple iteration for sklearn workflows
"""

import numpy as np
from bovi_core.ml.dataloaders.datasets.image_dataset import ImageDataset
from bovi_core.ml.dataloaders.loaders.sklearn_loader import SklearnDataLoader
from bovi_core.ml.dataloaders.sources.local_source import LocalFileSource
from PIL import Image

# Fixtures used from conftest:
# - image_dataset_large (from loaders/conftest.py)
# - mock_dataloader_config (from dataloaders/conftest.py)


class TestSklearnDataLoader:
    """Test SklearnDataLoader."""

    def test_loader_initialization(self, image_dataset_large, mock_dataloader_config):
        """Test loader initialization."""
        loader = SklearnDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=4,
        )

        assert loader.batch_size == 4
        assert loader.split == "train"
        assert loader.indices is not None

    def test_loader_uses_config_defaults(self, image_dataset_large, mock_dataloader_config):
        """Test loader uses config defaults."""
        loader = SklearnDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
        )

        # Should use config batch size
        assert loader.batch_size == 8

    def test_loader_length(self, image_dataset_large, mock_dataloader_config):
        """Test loader returns correct number of batches."""
        loader = SklearnDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train", batch_size=8
        )

        # 40 images / batch_size=8 = 5 batches
        assert len(loader) == 5
        assert loader.num_batches == 5
        assert loader.num_samples == 40

    def test_loader_iteration(self, image_dataset_large, mock_dataloader_config):
        """Test iterating over loader."""
        loader = SklearnDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train", batch_size=8
        )

        batches = list(loader)
        assert len(batches) == 5

        # Check batch structure
        batch = batches[0]
        assert "image" in batch
        assert "label" in batch

        # Check types (sklearn uses NumPy arrays)
        assert len(batch["label"]) == 8

    def test_loader_shuffle(self, image_dataset_large, mock_dataloader_config):
        """Test shuffle parameter."""
        # Train should shuffle by default
        loader_train = SklearnDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train", batch_size=8
        )
        assert loader_train.shuffle is True

        # Val should not shuffle by default
        loader_val = SklearnDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="val", batch_size=8
        )
        assert loader_val.shuffle is False

        # Can override
        loader_custom = SklearnDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="val",
            batch_size=8,
            shuffle=True,
        )
        assert loader_custom.shuffle is True

    def test_loader_get_sklearn_iterator(self, image_dataset_large, mock_dataloader_config):
        """Test get_sklearn_iterator returns iterator."""
        loader = SklearnDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train"
        )

        iterator = loader.get_sklearn_iterator()
        assert iterator is not None
        assert hasattr(iterator, "__iter__")

    def test_loader_get_pytorch_loader(self, image_dataset_large, mock_dataloader_config):
        """Test get_pytorch_loader returns None."""
        loader = SklearnDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train"
        )

        assert loader.get_pytorch_loader() is None

    def test_loader_get_tensorflow_dataset(self, image_dataset_large, mock_dataloader_config):
        """Test get_tensorflow_dataset returns None."""
        loader = SklearnDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train"
        )

        assert loader.get_tensorflow_dataset() is None

    def test_loader_multiple_epochs(self, image_dataset_large, mock_dataloader_config):
        """Test loader can iterate multiple epochs."""
        loader = SklearnDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train", batch_size=8
        )

        # First epoch
        epoch1_batches = list(loader)
        assert len(epoch1_batches) == 5

        # Second epoch
        epoch2_batches = list(loader)
        assert len(epoch2_batches) == 5

    def test_loader_get_all_data(self, image_dataset_large, mock_dataloader_config):
        """Test get_all_data method."""
        loader = SklearnDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train", batch_size=8
        )

        all_data = loader.get_all_data()

        assert "image" in all_data
        assert "label" in all_data
        assert len(all_data["label"]) == 40

    def test_loader_deterministic_shuffle(self, image_dataset_large, mock_dataloader_config):
        """Test shuffle is deterministic with same seed."""
        loader1 = SklearnDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            batch_size=8,
            shuffle=True,
            seed=42,
        )

        loader2 = SklearnDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            batch_size=8,
            shuffle=True,
            seed=42,
        )

        # Get first batches from each
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # Should be identical with same seed
        assert batch1["label"] == batch2["label"]

    def test_loader_empty_dataset(self, mock_dataloader_config, tmp_path):
        """Test loader with empty dataset."""
        # Create empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        source = LocalFileSource(empty_dir, file_pattern="*.jpg")
        dataset = ImageDataset(source)

        loader = SklearnDataLoader(
            dataset, config=mock_dataloader_config, split="val", batch_size=8
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
        dataset = ImageDataset(source)

        loader = SklearnDataLoader(
            dataset, config=mock_dataloader_config, split="train", batch_size=4
        )

        batches = list(loader)
        assert len(batches) == 1
        assert len(batches[0]["label"]) == 1
