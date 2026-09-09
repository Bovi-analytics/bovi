"""Tests for SklearnDataLoader.

Tests the NumPy-First architecture where:
- Datasets return raw NumPy arrays
- SklearnDataLoader provides simple iteration for sklearn workflows
"""

import numpy as np
import pytest
from bovi_core.ml.dataloaders.adapters import FrameworkAdapter
from bovi_core.ml.dataloaders.datasets.feature_vector_dataset import FeatureVectorDataset
from bovi_core.ml.dataloaders.datasets.image_dataset import ImageDataset
from bovi_core.ml.dataloaders.loaders.sklearn_loader import SklearnDataLoader
from bovi_core.ml.dataloaders.sources.dict_source import DictSource
from bovi_core.ml.dataloaders.sources.local_source import LocalFileSource
from PIL import Image

# Fixtures used from conftest:
# - image_dataset_large (from loaders/conftest.py)
# - mock_dataloader_config (from dataloaders/conftest.py)


def test_dense_batch_contract(dense_samples, mock_dataloader_config):
    loader = SklearnDataLoader(
        dense_samples, config=mock_dataloader_config, batch_size=2, shuffle=False
    )
    for _ in range(2):
        batches = list(loader)
        assert len(batches) == len(loader) == 2
        batch = batches[0]
        np.testing.assert_array_equal(batch["features"]["nested"]["vector"], [[0, 1], [1, 2]])
        np.testing.assert_array_equal(batch["features"]["sequence"], [[0, 2], [1, 3]])
        assert batch["features"]["pixels"].shape == (2, 2, 2, 3)
        assert batch["features"]["pixels"].dtype == np.uint8
        assert batch["labels"].dtype == np.float32
        assert batches[-1]["labels"].shape == (1,)
        assert batch["metadata"] == [sample["metadata"] for sample in dense_samples[:2]]


def test_numpy_collation_ignores_mapping_insertion_order():
    batch = FrameworkAdapter.numpy_collate([{"x": 1, "y": 2}, {"y": 4, "x": 3}])
    np.testing.assert_array_equal(batch["x"], [1, 3])


class _FeatureDataset(FeatureVectorDataset):
    def _get_features(self, raw_data):
        return {"milk": raw_data["milk"], "days": raw_data["days"]}

    def _get_labels(self, raw_data):
        return raw_data["target"]

    def _get_metadata(self, raw_data, index):
        return {"farm_id": raw_data["farm_id"], "index": index}


def test_seeded_epoch_stream_and_metrics_replay(shuffle_samples, mock_dataloader_config):
    loaders = [
        SklearnDataLoader(shuffle_samples, mock_dataloader_config, batch_size=7, seed=42)
        for _ in range(2)
    ]

    def order(loader):
        return np.concatenate([batch["features"] for batch in loader]).tolist()

    streams = [[order(loader) for _ in range(3)] for loader in loaders]
    assert streams[0] == streams[1]
    assert streams[0][0] != streams[0][1]
    loader = loaders[0]
    loader.set_epoch(1)
    assert order(loader) == order(loader) == streams[0][1]
    loader.set_epoch(2)
    assert order(loader) == streams[0][2]


def test_evaluation_iteration_is_stable(shuffle_samples, mock_dataloader_config):
    loader = SklearnDataLoader(shuffle_samples, mock_dataloader_config, split="validation")
    for _ in range(2):
        assert np.concatenate([batch["features"] for batch in loader]).tolist() == list(range(32))


@pytest.mark.parametrize("epoch", [-1, 1.5, True])
def test_invalid_epoch_rejected(epoch, shuffle_samples, mock_dataloader_config):
    loader = SklearnDataLoader(shuffle_samples, mock_dataloader_config)
    with pytest.raises(ValueError, match="nonnegative integer"):
        loader.set_epoch(epoch)


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

    def test_loader_iter_returns_batches(self, image_dataset_large, mock_dataloader_config):
        """Test loader iteration returns batches."""
        loader = SklearnDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train"
        )

        batch = next(iter(loader))
        assert "image" in batch
        assert "label" in batch

    def test_loader_multiple_epochs(self, image_dataset_large, mock_dataloader_config):
        """Test loader can iterate multiple epochs."""
        loader = SklearnDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train", batch_size=8
        )

        # First epoch
        epoch1_batches = list(loader)
        assert len(epoch1_batches) == 5
        first_order = loader.indices.copy()

        # Second epoch
        epoch2_batches = list(loader)
        assert len(epoch2_batches) == 5
        assert not np.array_equal(first_order, loader.indices)

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

    def test_nested_feature_mappings_are_collated(self, mock_dataloader_config):
        source = DictSource(
            [
                {"milk": 20.0, "days": 10, "target": 21.0, "farm_id": "farm-a"},
                {"milk": 30.0, "days": 20, "target": 29.0, "farm_id": "farm-b"},
            ]
        )
        loader = SklearnDataLoader(
            _FeatureDataset(source),
            config=mock_dataloader_config,
            batch_size=2,
            shuffle=False,
        )

        batch = next(iter(loader))

        np.testing.assert_array_equal(batch["features"]["milk"], np.array([20.0, 30.0]))
        np.testing.assert_array_equal(batch["features"]["days"], np.array([10, 20]))
        np.testing.assert_array_equal(batch["labels"], np.array([21.0, 29.0]))
        assert batch["metadata"] == [
            {"farm_id": "farm-a", "index": 0},
            {"farm_id": "farm-b", "index": 1},
        ]

    def test_numpy_collate_keeps_unstackable_values_as_lists(self):
        batch = [
            {"features": {"sequence": np.array([1.0, 2.0])}, "metadata": {"id": "a"}},
            {"features": {"sequence": np.array([3.0])}, "metadata": {"id": "b"}},
        ]

        collated = FrameworkAdapter.numpy_collate(batch)

        assert isinstance(collated["features"]["sequence"], list)
        assert collated["metadata"] == [{"id": "a"}, {"id": "b"}]
