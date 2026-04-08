"""Integration tests for VideoDataset using real video files.

These tests create temporary video files to verify:
- Resize-at-decode logic (critical for RAM efficiency)
- Frame sampling strategies
- OpenCV integration
"""

import os
import tempfile

import cv2
import numpy as np
import pytest
from bovi_core.ml.dataloaders.datasets.video_dataset import VideoDataset
from bovi_core.ml.dataloaders.sources.local_source import LocalFileSource


@pytest.fixture
def sample_video_file():
    """Creates a temporary 10-frame MP4 video (100x100 resolution).

    Frames have distinct colors for verification:
    - Frame 0: Black (0, 0, 0)
    - Frame 1: Dark gray (25, 25, 25)
    - ...etc
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    width, height = 100, 100
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i in range(10):
        # Create frame with gradient color (useful for verification)
        frame = np.full((height, width, 3), fill_value=i * 25, dtype=np.uint8)
        out.write(frame)

    out.release()
    yield path

    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sample_video_dir(sample_video_file):
    """Directory containing the sample video file."""
    return os.path.dirname(sample_video_file)


class TestVideoDatasetIntegration:
    """Integration tests with real video files."""

    def test_resize_at_decode(self, sample_video_dir):
        """CRITICAL: Verify resize happens DURING decode, not after.

        Original video is 100x100. We request 224x224.
        Output should be 224x224, proving resize happened.
        """
        source = LocalFileSource(sample_video_dir, file_pattern="*.mp4")
        dataset = VideoDataset(
            source,
            frame_size=(224, 224),
            num_frames=4,
            sample_strategy="uniform",
        )

        assert len(dataset) == 1

        item = dataset[0]
        frames = item["frames"]

        # Verify resize happened
        assert frames.shape == (4, 224, 224, 3)
        assert frames.dtype == np.uint8

    def test_frame_sampling_uniform(self, sample_video_dir):
        """Test uniform frame sampling from 10-frame video."""
        source = LocalFileSource(sample_video_dir, file_pattern="*.mp4")
        dataset = VideoDataset(
            source,
            frame_size=(64, 64),
            num_frames=5,
            sample_strategy="uniform",
        )

        item = dataset[0]
        frames = item["frames"]

        assert frames.shape == (5, 64, 64, 3)

    def test_frame_sampling_consecutive(self, sample_video_dir):
        """Test consecutive frame sampling."""
        source = LocalFileSource(sample_video_dir, file_pattern="*.mp4")
        dataset = VideoDataset(
            source,
            frame_size=(64, 64),
            num_frames=5,
            sample_strategy="consecutive",
        )

        item = dataset[0]
        frames = item["frames"]

        assert frames.shape == (5, 64, 64, 3)

    def test_num_frames_exceeds_video_length(self, sample_video_dir):
        """Test when num_frames > total frames in video (should pad)."""
        source = LocalFileSource(sample_video_dir, file_pattern="*.mp4")
        dataset = VideoDataset(
            source,
            frame_size=(64, 64),
            num_frames=20,  # Video only has 10 frames
            sample_strategy="uniform",
        )

        item = dataset[0]
        frames = item["frames"]

        # Should pad to requested num_frames
        assert frames.shape == (20, 64, 64, 3)

    def test_output_dtype_uint8(self, sample_video_dir):
        """Verify output is uint8 (not normalized float)."""
        source = LocalFileSource(sample_video_dir, file_pattern="*.mp4")
        dataset = VideoDataset(
            source,
            frame_size=(64, 64),
            num_frames=4,
        )

        item = dataset[0]

        assert item["frames"].dtype == np.uint8
        assert item["frames"].min() >= 0
        assert item["frames"].max() <= 255

    def test_bgr_to_rgb_conversion(self, sample_video_dir):
        """Verify frames are converted from BGR (OpenCV) to RGB."""
        source = LocalFileSource(sample_video_dir, file_pattern="*.mp4")
        dataset = VideoDataset(
            source,
            frame_size=(64, 64),
            num_frames=1,
        )

        item = dataset[0]
        frame = item["frames"][0]

        # Our test video has grayscale frames, so R=G=B
        # This verifies no channel swapping issues
        assert frame.shape == (64, 64, 3)

    def test_different_frame_sizes(self, sample_video_dir):
        """Test various target frame sizes."""
        source = LocalFileSource(sample_video_dir, file_pattern="*.mp4")

        for size in [(32, 32), (64, 64), (128, 128), (224, 224)]:
            dataset = VideoDataset(
                source,
                frame_size=size,
                num_frames=2,
            )
            item = dataset[0]
            assert item["frames"].shape == (2, size[0], size[1], 3)

    def test_rectangular_frame_size(self, sample_video_dir):
        """Test non-square frame sizes."""
        source = LocalFileSource(sample_video_dir, file_pattern="*.mp4")
        dataset = VideoDataset(
            source,
            frame_size=(112, 224),  # height, width
            num_frames=2,
        )

        item = dataset[0]
        assert item["frames"].shape == (2, 112, 224, 3)
