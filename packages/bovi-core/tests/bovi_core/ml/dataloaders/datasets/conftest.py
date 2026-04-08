"""Fixtures specific to dataset tests."""

import numpy as np
import pytest
from bovi_core.ml.dataloaders.base import DataSource
from bovi_core.ml.dataloaders.datasets.image_dataset import ImageDataset


@pytest.fixture
def image_dataset(image_source):
    """ImageDataset with small source (3 images)."""
    return ImageDataset(image_source)


class MockVideoSource(DataSource):
    """Mock video data source for testing."""

    def __init__(self, num_videos: int = 3) -> None:
        self.num_videos = num_videos
        self.video_data = {i: b"fake_video_bytes" for i in range(num_videos)}

    def __len__(self) -> int:
        return self.num_videos

    def load_item(self, key: int) -> bytes:
        return self.video_data[key]

    def get_metadata(self, key: int) -> dict[str, object]:
        labels = ["action1", "action2", "action1"]
        return {
            "path": f"video_{key}.mp4",
            "label": labels[key % len(labels)],
            "index": key,
        }

    def get_keys(self) -> list[int]:
        return list(range(self.num_videos))


@pytest.fixture
def mock_video_source():
    """MockVideoSource instance with 3 videos."""
    return MockVideoSource(num_videos=3)


@pytest.fixture
def mock_numpy_frames():
    """Create mock NumPy video frames (32 frames of 64x64)."""
    frames = []
    for _ in range(32):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        frames.append(frame)
    return frames
