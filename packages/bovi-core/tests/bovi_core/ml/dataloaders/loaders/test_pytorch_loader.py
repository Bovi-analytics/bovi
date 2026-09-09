"""Tests for PyTorchDataLoader.

Tests the NumPy-First architecture where:
- Datasets return raw NumPy arrays (no transforms)
- Sample transforms are explicit on TransformedDataset, before batching
- Albumentations fields are selected explicitly by a sample transform
"""

import numpy as np
import pytest
from bovi_core.ml.dataloaders.datasets import TransformedDataset
from bovi_core.ml.dataloaders.datasets.image_dataset import ImageDataset
from bovi_core.ml.dataloaders.loaders.pytorch_loader import PyTorchDataLoader
from bovi_core.ml.dataloaders.sources.local_source import LocalFileSource
from bovi_core.ml.dataloaders.transforms import AlbumentationsTransform, ImagePreprocessing
from PIL import Image

torch = pytest.importorskip("torch", reason="PyTorch is required for PyTorchDataLoader tests")

pytestmark = [pytest.mark.core, pytest.mark.torch]


@pytest.mark.parametrize("workers", [0, 1])
def test_seeded_epoch_stream_and_metrics_replay(workers, shuffle_samples, mock_dataloader_config):
    loaders = [
        PyTorchDataLoader(
            shuffle_samples,
            mock_dataloader_config,
            batch_size=7,
            seed=42,
            num_workers=workers,
            persistent_workers=workers > 0,
        )
        for _ in range(2)
    ]

    def order(loader):
        return torch.cat([batch["features"] for batch in loader]).tolist()

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


def test_generator_policy_and_unseeded_epoch_error(shuffle_samples, mock_dataloader_config):
    generator = torch.Generator().manual_seed(7)
    loader = PyTorchDataLoader(
        shuffle_samples, mock_dataloader_config, generator=generator, num_workers=0
    )
    assert loader.generator is generator
    loader.set_epoch(2)
    with pytest.raises(ValueError, match="not both"):
        PyTorchDataLoader(shuffle_samples, mock_dataloader_config, seed=7, generator=generator)
    unseeded = PyTorchDataLoader(shuffle_samples, mock_dataloader_config, num_workers=0)
    with pytest.raises(ValueError, match="requires seed"):
        unseeded.set_epoch(0)


def test_evaluation_iteration_is_stable(shuffle_samples, mock_dataloader_config):
    loader = PyTorchDataLoader(
        shuffle_samples, mock_dataloader_config, split="validation", num_workers=0, seed=42
    )
    for _ in range(2):
        assert torch.cat([batch["features"] for batch in loader]).tolist() == list(range(32))


@pytest.mark.parametrize("epoch", [-1, 1.5, True])
def test_invalid_epoch_rejected(epoch, shuffle_samples, mock_dataloader_config):
    loader = PyTorchDataLoader(shuffle_samples, mock_dataloader_config, seed=42, num_workers=0)
    with pytest.raises(ValueError, match="nonnegative integer"):
        loader.set_epoch(epoch)


def test_worker_hook_is_forwarded(shuffle_samples, mock_dataloader_config):
    def initialize(worker_id):
        pass

    loader = PyTorchDataLoader(
        shuffle_samples, mock_dataloader_config, seed=42, num_workers=0, worker_init_fn=initialize
    )
    assert loader._pytorch_loader is not None
    assert loader._pytorch_loader.worker_init_fn is initialize


def test_dense_batch_contract(dense_samples, mock_dataloader_config):
    loader = PyTorchDataLoader(
        dense_samples, config=mock_dataloader_config, batch_size=2, shuffle=False, num_workers=0
    )
    for _ in range(2):
        batches = list(loader)
        assert len(batches) == len(loader) == 2
        batch = batches[0]
        np.testing.assert_array_equal(batch["features"]["nested"]["vector"], [[0, 1], [1, 2]])
        assert batch["features"]["nested"]["vector"].dtype == torch.float64
        np.testing.assert_array_equal(batch["features"]["sequence"], [[0, 2], [1, 3]])
        assert batch["features"]["pixels"].shape == (2, 2, 2, 3)
        assert batch["features"]["pixels"].dtype == torch.uint8
        assert batch["features"]["enabled"].dtype == torch.bool
        assert batch["labels"].dtype == torch.float32
        np.testing.assert_array_equal(batch["labels"], [0.5, 1.5])
        assert batches[-1]["labels"].shape == (1,)
        assert batch["metadata"] == [sample["metadata"] for sample in dense_samples[:2]]


def test_custom_collator_replaces_adapter(dense_samples, mock_dataloader_config):
    loader = PyTorchDataLoader(
        dense_samples,
        config=mock_dataloader_config,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )
    assert next(iter(loader)) == dense_samples[:2]


def test_metadata_and_ragged_values_have_explicit_policy():
    from bovi_core.ml.dataloaders.batching import collate_pytorch_samples

    samples = [
        {"features": np.arange(2), "metadata": {"index": 0, "opaque": None}},
        {"features": np.arange(3), "metadata": {"index": 1, "opaque": object()}},
    ]
    batch = collate_pytorch_samples(samples)
    assert isinstance(batch["features"], list)
    assert batch["metadata"][1] is samples[1]["metadata"]
    columns = collate_pytorch_samples(samples, preserve_keys=())
    assert torch.equal(columns["metadata"]["index"], torch.tensor([0, 1]))


@pytest.mark.parametrize("vision", [True, False])
def test_video_conversion_is_explicit(vision):
    from bovi_core.ml.dataloaders.batching import collate_pytorch_samples

    frames = np.full((2, 4, 5, 3), 255, dtype=np.uint8)
    batch = collate_pytorch_samples(
        [
            ImagePreprocessing(fields=("frames",), normalize=vision, channels_first=vision)(
                {"frames": frames}
            )
        ]
    )
    assert batch["frames"].shape == ((1, 2, 3, 4, 5) if vision else (1, 2, 4, 5, 3))
    assert batch["frames"].dtype == (torch.float32 if vision else torch.uint8)
    assert batch["frames"].flatten()[0].item() == (1 if vision else 255)


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
        assert batch["image"].shape[-1] == 3  # Layout is unchanged
        assert batch["image"].dtype == torch.uint8  # Values are unchanged

    def test_loader_iteration_with_albumentations_transform(
        self, image_dataset_large, mock_dataloader_config, albumentations_resize_transform
    ):
        """Test iterating over loader WITH Albumentations transform."""
        loader = PyTorchDataLoader(
            TransformedDataset(
                image_dataset_large,
                [
                    AlbumentationsTransform(albumentations_resize_transform),
                    ImagePreprocessing(normalize=True, channels_first=True),
                ],
            ),
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

    @pytest.mark.multiprocessing
    @pytest.mark.skip(reason="Multiprocessing DataLoader iteration is flaky in CI/sandbox runners")
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

    def test_loader_iter_returns_batches(self, image_dataset_large, mock_dataloader_config):
        """Test loader iteration returns batches."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            num_workers=0,
        )

        batch = next(iter(loader))
        assert "image" in batch
        assert "label" in batch

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

    @pytest.mark.multiprocessing
    @pytest.mark.skip(reason="Multiprocessing DataLoader iteration is flaky in CI/sandbox runners")
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

    def test_loader_preserves_image_layout(self, image_dataset_large, mock_dataloader_config):
        """A loader does not choose a model-specific image layout."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=4,
            num_workers=0,
        )

        batch = next(iter(loader))

        # Without preprocessing, images stay in (B, H, W, C) format
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape[0] == 4  # Batch size
        assert batch["image"].shape[-1] == 3  # Channels last (HWC)

    def test_loader_preserves_image_dtype(self, image_dataset_large, mock_dataloader_config):
        """A loader does not normalize pixel values."""
        loader = PyTorchDataLoader(
            image_dataset_large,
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=4,
            num_workers=0,
        )

        batch = next(iter(loader))

        # Without preprocessing, images stay as uint8
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].dtype == torch.uint8

    def test_loader_transform_parameter(
        self, image_dataset_large, mock_dataloader_config, albumentations_resize_transform
    ):
        """Test that preprocessing belongs to the wrapped dataset."""
        loader = PyTorchDataLoader(
            TransformedDataset(
                image_dataset_large,
                [
                    AlbumentationsTransform(albumentations_resize_transform),
                    ImagePreprocessing(normalize=True, channels_first=True),
                ],
            ),
            config=mock_dataloader_config,
            split="train",
            model_name="test_model",
            batch_size=4,
            num_workers=0,
        )

        # The loader only receives a dataset; preprocessing is explicit on its wrapper.
        assert isinstance(loader.dataset, TransformedDataset)
        transform = loader.dataset.transforms[0]
        assert isinstance(transform, AlbumentationsTransform)
        assert transform.pipeline is albumentations_resize_transform
