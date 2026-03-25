"""Tests for VideoDataset.

Tests the NumPy-First architecture where:
- Datasets return raw NumPy arrays (no transforms)
- Videos are resized DURING decode to prevent RAM explosion
- Transforms are applied in DataLoaders via FrameworkAdapter
"""

from unittest.mock import patch

import numpy as np
import pytest

from bovi_core.ml.dataloaders.datasets.video_dataset import VideoDataset

# MockVideoSource is imported from conftest.py


class TestVideoDataset:
    """Test VideoDataset with NumPy-First architecture."""

    def test_dataset_initialization(self, mock_video_source):
        """Test dataset initialization."""
        dataset = VideoDataset(mock_video_source, num_frames=16, sample_strategy="uniform")

        assert len(dataset) == 3
        assert dataset.num_frames == 16
        assert dataset.sample_strategy == "uniform"

    def test_dataset_frame_size_parameter(self, mock_video_source):
        """Test that frame_size parameter is accepted (resize-at-decode)."""
        dataset = VideoDataset(mock_video_source, frame_size=(224, 224))

        assert dataset.frame_size == (224, 224)

    def test_dataset_invalid_strategy(self, mock_video_source):
        """Test invalid sampling strategy raises error."""
        with pytest.raises(ValueError, match="Invalid sample_strategy"):
            VideoDataset(mock_video_source, sample_strategy="invalid")

    @patch("bovi_core.ml.dataloaders.datasets.video_dataset.VideoDataset._extract_frames")
    def test_dataset_getitem_returns_numpy(self, mock_extract, mock_video_source, mock_numpy_frames):
        """Test that getitem returns NumPy arrays (not PIL)."""
        # Stack frames into (T, H, W, C) array
        mock_extract.return_value = np.stack(mock_numpy_frames[:16], axis=0)

        dataset = VideoDataset(mock_video_source, num_frames=16)
        item = dataset[0]

        assert "frames" in item
        assert "label" in item
        # NumPy-First: Returns NumPy array (T, H, W, C)
        assert isinstance(item["frames"], np.ndarray)
        assert item["frames"].shape == (16, 64, 64, 3)
        assert item["frames"].dtype == np.uint8

    @patch("bovi_core.ml.dataloaders.datasets.video_dataset.VideoDataset._extract_frames")
    def test_dataset_getitem_with_metadata(self, mock_extract, mock_video_source, mock_numpy_frames):
        """Test getitem with metadata."""
        mock_extract.return_value = np.stack(mock_numpy_frames[:16], axis=0)

        dataset = VideoDataset(mock_video_source, num_frames=16, return_metadata=True)
        item = dataset[0]

        assert "frames" in item
        assert "label" in item
        assert "metadata" in item
        assert "path" in item["metadata"]

    @patch("bovi_core.ml.dataloaders.datasets.video_dataset.VideoDataset._extract_frames")
    def test_dataset_different_num_frames(self, mock_extract, mock_video_source, mock_numpy_frames):
        """Test with different num_frames settings."""
        for num_frames in [4, 8, 16, 32]:
            mock_extract.return_value = np.stack(mock_numpy_frames[:num_frames], axis=0)

            dataset = VideoDataset(mock_video_source, num_frames=num_frames)
            item = dataset[0]

            assert item["frames"].shape[0] == num_frames

    @patch("bovi_core.ml.dataloaders.datasets.video_dataset.VideoDataset._extract_frames")
    def test_dataset_get_sample(self, mock_extract, mock_video_source, mock_numpy_frames):
        """Test get_sample method."""
        mock_extract.return_value = np.stack(mock_numpy_frames[:16], axis=0)

        dataset = VideoDataset(mock_video_source, num_frames=16)
        sample = dataset.get_sample(0)

        assert "frames" in sample
        assert "label" in sample
        assert isinstance(sample["frames"], np.ndarray)

    @patch("bovi_core.ml.dataloaders.datasets.video_dataset.VideoDataset._extract_frames")
    def test_dataset_iteration(self, mock_extract, mock_video_source, mock_numpy_frames):
        """Test iterating over dataset."""
        mock_extract.return_value = np.stack(mock_numpy_frames[:16], axis=0)

        dataset = VideoDataset(mock_video_source, num_frames=16)

        # Manually iterate
        items = []
        for i in range(len(dataset)):
            items.append(dataset[i])

        assert len(items) == 3
        for item in items:
            assert "frames" in item
            assert "label" in item
            assert isinstance(item["frames"], np.ndarray)

    def test_sampling_strategies_accepted(self, mock_video_source):
        """Test different frame sampling strategies are accepted."""
        for strategy in ["uniform", "random", "consecutive"]:
            dataset = VideoDataset(mock_video_source, sample_strategy=strategy)
            assert dataset.sample_strategy == strategy

    def test_dataset_no_transform_argument(self, mock_video_source):
        """Test that Dataset does not accept transform argument (NumPy-First)."""
        # VideoDataset should NOT have a transform parameter
        dataset = VideoDataset(mock_video_source)
        assert not hasattr(dataset, "transform") or dataset.__dict__.get("transform") is None

    @patch("bovi_core.ml.dataloaders.datasets.video_dataset.VideoDataset._extract_frames")
    def test_dataset_returns_uint8(self, mock_extract, mock_video_source, mock_numpy_frames):
        """Test that dataset returns uint8 frames (no normalization)."""
        mock_extract.return_value = np.stack(mock_numpy_frames[:16], axis=0)

        dataset = VideoDataset(mock_video_source, num_frames=16)
        item = dataset[0]

        # Raw NumPy, no normalization
        assert item["frames"].dtype == np.uint8
        assert item["frames"].min() >= 0
        assert item["frames"].max() <= 255

    @patch("bovi_core.ml.dataloaders.datasets.video_dataset.VideoDataset._extract_frames")
    def test_frame_size_applied_in_output(self, mock_extract, mock_video_source):
        """Test that frame_size is respected in output."""
        # Simulate resized frames (224x224)
        resized_frames = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
        mock_extract.return_value = resized_frames

        dataset = VideoDataset(mock_video_source, num_frames=16, frame_size=(224, 224))
        item = dataset[0]

        assert item["frames"].shape == (16, 224, 224, 3)
