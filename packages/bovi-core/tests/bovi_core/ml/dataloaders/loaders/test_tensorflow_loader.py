"""Tests for TensorFlowDataLoader.

Tests the NumPy-First architecture where:
- Datasets return raw NumPy arrays (no transforms)
- Transforms are applied explicitly on TransformedDataset
- Albumentations fields are selected explicitly by a sample transform
"""

from typing import Any

import numpy as np

# Fixtures used from conftest:
# - image_dataset_large (from loaders/conftest.py)
# - mock_dataloader_config (from dataloaders/conftest.py)
# - albumentations_resize_transform (from loaders/conftest.py)
import pytest
from bovi_core.ml.dataloaders.datasets import TransformedDataset
from bovi_core.ml.dataloaders.datasets.base_dataset import Dataset
from bovi_core.ml.dataloaders.datasets.image_dataset import ImageDataset
from bovi_core.ml.dataloaders.loaders.tensorflow_loader import TensorFlowDataLoader
from bovi_core.ml.dataloaders.sources.dict_source import DictSource
from bovi_core.ml.dataloaders.sources.local_source import LocalFileSource
from bovi_core.ml.dataloaders.transforms import AlbumentationsTransform, ImagePreprocessing
from PIL import Image

tf = pytest.importorskip(
    "tensorflow", reason="TensorFlow is required for TensorFlowDataLoader tests"
)

pytestmark = [pytest.mark.core, pytest.mark.tensorflow]


class _SampleDataset(Dataset):
    """Serve test records unchanged and count accesses for cache assertions."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        super().__init__(DictSource(samples))
        self.reads = 0

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.reads += 1
        return self.source.load_item(index)


def test_seeded_epoch_stream_and_metrics_replay(shuffle_samples, mock_dataloader_config):
    loaders = [
        TensorFlowDataLoader(
            _SampleDataset(shuffle_samples), mock_dataloader_config, batch_size=7, seed=42
        )
        for _ in range(2)
    ]

    def order(loader):
        return np.concatenate([batch["features"] for batch in loader]).tolist()

    streams = [[order(loader) for _ in range(3)] for loader in loaders]
    assert streams[0] == streams[1]
    assert streams[0][0] != streams[0][1]
    for loader in loaders:
        loader.set_epoch(5)
    expected = order(loaders[0])
    assert order(loaders[0]) == order(loaders[1]) == expected
    loaders[0].set_epoch(6)
    assert order(loaders[0]) != expected
    loaders[0].set_epoch(5)
    assert order(loaders[0]) == expected


def test_explicit_replay_and_unseeded_epoch_error(shuffle_samples, mock_dataloader_config):
    loader = TensorFlowDataLoader(
        _SampleDataset(shuffle_samples),
        mock_dataloader_config,
        seed=7,
        reshuffle_each_iteration=False,
    )
    first = np.concatenate([batch["features"] for batch in loader])
    np.testing.assert_array_equal(first, np.concatenate([batch["features"] for batch in loader]))
    unseeded = TensorFlowDataLoader(_SampleDataset(shuffle_samples), mock_dataloader_config)
    with pytest.raises(ValueError, match="requires seed"):
        unseeded.set_epoch(0)


def test_evaluation_iteration_is_stable(shuffle_samples, mock_dataloader_config):
    loader = TensorFlowDataLoader(
        _SampleDataset(shuffle_samples), mock_dataloader_config, split="validation"
    )
    for _ in range(2):
        assert np.concatenate([batch["features"] for batch in loader]).tolist() == list(range(32))


@pytest.mark.parametrize("epoch", [-1, 1.5, True])
def test_invalid_epoch_rejected(epoch, shuffle_samples, mock_dataloader_config):
    loader = TensorFlowDataLoader(_SampleDataset(shuffle_samples), mock_dataloader_config, seed=42)
    with pytest.raises(ValueError, match="nonnegative integer"):
        loader.set_epoch(epoch)


def test_epoch_change_preserves_cached_samples(shuffle_samples, mock_dataloader_config):
    samples = _SampleDataset(shuffle_samples)
    loader = TensorFlowDataLoader(samples, mock_dataloader_config, seed=42, cache=True)
    first = np.concatenate([batch["features"] for batch in loader])
    reads = samples.reads
    loader.set_epoch(1)
    second = np.concatenate([batch["features"] for batch in loader])
    assert samples.reads == reads
    np.testing.assert_array_equal(np.sort(first), np.arange(32))
    np.testing.assert_array_equal(np.sort(second), np.sort(first))


def test_dense_batch_contract(dense_samples, mock_dataloader_config):
    loader = TensorFlowDataLoader(
        _SampleDataset(dense_samples), config=mock_dataloader_config, batch_size=2, shuffle=False
    )
    for _ in range(2):
        batches = list(loader)
        assert len(batches) == len(loader) == 2
        batch = batches[0]
        np.testing.assert_array_equal(batch["features"]["nested"]["vector"], [[0, 1], [1, 2]])
        assert batch["features"]["nested"]["vector"].dtype == tf.float64
        np.testing.assert_array_equal(batch["features"]["sequence"], [[0, 2], [1, 3]])
        assert batch["features"]["pixels"].shape == (2, 2, 2, 3)
        assert batch["features"]["pixels"].dtype == tf.uint8
        assert batch["features"]["enabled"].dtype == tf.bool
        assert batch["labels"].dtype == tf.float32
        np.testing.assert_array_equal(batch["labels"], [0.5, 1.5])
        assert batches[-1]["labels"].shape == (1,)
        np.testing.assert_array_equal(batch["metadata"]["id"], [b"0", b"1"])
        np.testing.assert_array_equal(batch["metadata"]["nested"]["index"], [0, 1])


@pytest.mark.parametrize("unsupported", [None, object()])
def test_unsupported_metadata_rejected_or_explicitly_dropped(unsupported, mock_dataloader_config):
    samples = _SampleDataset([{"features": np.array([1.0]), "metadata": {"opaque": unsupported}}])
    with pytest.raises(TypeError, match=r"metadata\.opaque.*drop_keys"):
        TensorFlowDataLoader(samples, config=mock_dataloader_config)
    loader = TensorFlowDataLoader(samples, config=mock_dataloader_config, drop_keys=("metadata",))
    assert set(next(iter(loader))) == {"features"}


def test_explicit_signature_supports_variable_shapes_and_empty_data(mock_dataloader_config):
    signature = {"features": {"x": tf.TensorSpec((None,), tf.float32)}}
    samples = _SampleDataset([{"features": {"x": np.ones(n, dtype=np.float32)}} for n in (2, 3)])
    loader = TensorFlowDataLoader(
        samples,
        config=mock_dataloader_config,
        batch_size=1,
        shuffle=False,
        output_signature=signature,
    )
    assert [batch["features"]["x"].shape for batch in loader] == [(1, 2), (1, 3)]
    empty = TensorFlowDataLoader(
        _SampleDataset([]), config=mock_dataloader_config, output_signature=signature
    )
    assert list(empty) == []
    assert empty.element_spec is not None
    assert empty.element_spec["features"]["x"].shape == (None, None)


@pytest.mark.parametrize("prefetch", [None, 0, 3])
def test_prefetch_setting_reaches_dataset(prefetch, monkeypatch, mock_dataloader_config):
    calls = []
    original = tf.data.Dataset.prefetch

    def record_prefetch(dataset, buffer_size, *args, **kwargs):
        calls.append(buffer_size)
        return original(dataset, buffer_size, *args, **kwargs)

    monkeypatch.setattr(tf.data.Dataset, "prefetch", record_prefetch)
    TensorFlowDataLoader(
        _SampleDataset([{"features": np.ones(2)}]),
        config=mock_dataloader_config,
        prefetch_buffer_size=prefetch,
    )
    assert calls == [tf.data.AUTOTUNE if prefetch is None else prefetch]


@pytest.mark.parametrize("normalize", [True, False])
def test_transformed_image_normalization_option(normalize, mock_dataloader_config):
    samples = _SampleDataset([{"image": np.full((4, 5, 3), 255, dtype=np.uint8)}])
    loader = TensorFlowDataLoader(
        TransformedDataset(samples, [ImagePreprocessing(normalize=normalize)]),
        config=mock_dataloader_config,
    )
    image = next(iter(loader))["image"]
    assert image.shape == (1, 4, 5, 3)
    assert image.dtype == (tf.float32 if normalize else tf.uint8)
    assert image.numpy().flat[0] == (1 if normalize else 255)


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
            TransformedDataset(
                image_dataset_large,
                [
                    AlbumentationsTransform(albumentations_resize_transform),
                    ImagePreprocessing(normalize=True, channels_first=False),
                ],
            ),
            config=mock_dataloader_config,
            split="train",
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
        # The explicit image preprocessing normalizes uint8 to float32
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

    def test_loader_iter_returns_batches(self, image_dataset_large, mock_dataloader_config):
        """Test loader iteration returns batches."""

        loader = TensorFlowDataLoader(
            image_dataset_large, config=mock_dataloader_config, split="train"
        )

        batch = next(iter(loader))
        assert "image" in batch
        assert "label" in batch

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
        """Test that preprocessing belongs to the wrapped dataset."""
        loader = TensorFlowDataLoader(
            TransformedDataset(
                image_dataset_large,
                [
                    AlbumentationsTransform(albumentations_resize_transform),
                    ImagePreprocessing(normalize=True, channels_first=False),
                ],
            ),
            config=mock_dataloader_config,
            split="train",
            batch_size=4,
        )

        # The loader only receives a dataset; preprocessing is explicit on its wrapper.
        assert isinstance(loader.dataset, TransformedDataset)
        transform = loader.dataset.transforms[0]
        assert isinstance(transform, AlbumentationsTransform)
        assert transform.pipeline is albumentations_resize_transform

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
            TransformedDataset(
                image_dataset_large,
                [
                    AlbumentationsTransform(albumentations_resize_transform),
                    ImagePreprocessing(normalize=True, channels_first=False),
                ],
            ),
            config=mock_dataloader_config,
            split="train",
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
