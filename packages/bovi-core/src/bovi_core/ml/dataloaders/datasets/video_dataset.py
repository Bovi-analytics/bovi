"""
Video dataset implementation.

Resizes frames DURING decode to prevent RAM explosion.
Returns raw NumPy arrays - transforms are applied in DataLoaders.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from ..base import Dataset, DataSource

if TYPE_CHECKING:
    from bovi_core.config import Config

logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    """
    Dataset for video classification/action recognition tasks.

    Loads videos from a DataSource, extracts and resizes frames during decode,
    and returns NumPy arrays.

    IMPORTANT: This dataset resizes frames DURING the decode loop to prevent
    RAM explosion from storing full-resolution frames.

    Args:
        source: DataSource to load videos from.
        config: Optional config instance.
        num_frames: Number of frames to extract per video (default: 16).
        frame_size: Target frame size as (height, width) (default: (224, 224)).
        sample_strategy: How to sample frames - 'uniform', 'random', 'consecutive'.
        return_metadata: Whether to include metadata in output (default: False).

    Returns:
        Dict with keys:
        - "frames": NumPy array of shape (T, H, W, C), uint8
        - "label": Label (if available in metadata)
        - "metadata": Additional metadata (if return_metadata=True)
    """

    # Type annotations for instance attributes
    num_frames: int
    frame_size: tuple[int, int]
    sample_strategy: str
    return_metadata: bool

    def __init__(
        self,
        source: DataSource,
        config: Config | None = None,
        num_frames: int = 16,
        frame_size: tuple[int, int] = (224, 224),
        sample_strategy: str = "uniform",
        return_metadata: bool = False,
    ) -> None:
        super().__init__(source, config)
        self.num_frames = num_frames
        self.frame_size = frame_size  # (height, width)
        self.sample_strategy = sample_strategy
        self.return_metadata = return_metadata

        if sample_strategy not in ["uniform", "random", "consecutive"]:
            raise ValueError(
                f"Invalid sample_strategy: {sample_strategy}. "
                "Must be one of: uniform, random, consecutive"
            )

        logger.info(
            f"VideoDataset: {len(self.source)} videos, "
            f"num_frames={num_frames}, "
            f"frame_size={frame_size}, "
            f"sample_strategy={sample_strategy}"
        )

    def __len__(self) -> int:
        """Number of videos in dataset."""
        return len(self.source)

    def _extract_frames(self, video_bytes: bytes) -> np.ndarray:
        """
        Extract and resize frames from video bytes.

        CRITICAL: Resizes DURING the decode loop to prevent RAM explosion.
        This is the key fix for the video memory leak.

        Args:
            video_bytes: Raw video file bytes.

        Returns:
            NumPy array of shape (T, H, W, C), uint8.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV (cv2) is required for video processing. "
                "Install with: pip install opencv-python"
            )

        # Write bytes to temporary file for cv2
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                logger.warning("Video has 0 frames, returning empty array")
                cap.release()
                return np.zeros(
                    (self.num_frames, self.frame_size[0], self.frame_size[1], 3),
                    dtype=np.uint8,
                )

            # Sample frame indices based on strategy
            indices: list[int]
            if self.sample_strategy == "uniform":
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
            elif self.sample_strategy == "random":
                import random

                indices = sorted(
                    random.sample(range(total_frames), min(self.num_frames, total_frames))
                )
            else:  # consecutive
                indices = list(range(min(self.num_frames, total_frames)))

            # Extract and resize frames DURING the loop
            frames: list[np.ndarray] = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # RESIZE HERE - before appending! This prevents RAM explosion.
                    frame_small = cv2.resize(frame_rgb, (self.frame_size[1], self.frame_size[0]))
                    frames.append(frame_small)

            cap.release()

            # Pad with last frame if needed
            while len(frames) < self.num_frames:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(
                        np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
                    )

            # Stack into (T, H, W, C) array
            return np.stack(frames[: self.num_frames], axis=0)

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    def __getitem__(self, index: int) -> dict[str, object]:
        """
        Get video by index.

        Args:
            index: Video index.

        Returns:
            Dict with frames (NumPy THWC uint8), label, and optionally metadata.
        """
        # Load raw video bytes
        video_bytes = self.source.load_item(index)

        # Extract and resize frames
        frames = self._extract_frames(video_bytes)

        # Get metadata (for label)
        metadata = self.source.get_metadata(index)

        # Build output
        output: dict[str, object] = {
            "frames": frames,
            "label": metadata.get("label", None),
        }

        if self.return_metadata:
            output["metadata"] = metadata

        return output

    def get_sample(self, index: int = 0) -> dict[str, object]:
        """
        Get a sample video for inspection.

        Args:
            index: Index to sample (default: 0).

        Returns:
            Sample item dict.
        """
        return self[index]
