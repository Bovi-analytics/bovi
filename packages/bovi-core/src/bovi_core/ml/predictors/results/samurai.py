"""Samurai (video segmentation) prediction result.

Placeholder for future Samurai integration.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base import HumanReadablePredictionResult


class SamuraiPredictionResult(HumanReadablePredictionResult):
    """
    Samurai (video segmentation) prediction result.

    Placeholder for future implementation when Samurai predictor is activated.
    Will handle video frame segmentation with temporal consistency.

    Note: Samurai requires prompts (typically from a detector) to initialize tracking.
    """

    def __init__(
        self,
        masks: Optional[List[np.ndarray]] = None,
        scores: Optional[np.ndarray] = None,
        frame_ids: Optional[List[int]] = None,
        track_ids: Optional[List[int]] = None,
        original_frames: Optional[List[np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Samurai prediction result.

        Args:
            masks: Segmentation masks for each frame
            scores: Confidence scores
            frame_ids: Frame indices
            track_ids: Object tracking IDs
            original_frames: Original video frames
            metadata: Additional metadata (should include prompt info)
        """
        super().__init__(metadata)
        self.masks = masks
        self.scores = scores
        self.frame_ids = frame_ids
        self.track_ids = track_ids
        self.original_frames = original_frames

    @classmethod
    def from_raw(cls, raw_output: Any, **kwargs):
        """
        Create from Samurai output (to be implemented when Samurai predictor is activated).

        Args:
            raw_output: Samurai model output
            **kwargs: Additional context (must include 'prompt' for Samurai)

        Returns:
            SamuraiPredictionResult instance

        Raises:
            NotImplementedError: Samurai support is not yet implemented
        """
        raise NotImplementedError(
            "Samurai support is planned for future release. "
            "The Samurai predictor is currently commented out in the codebase. "
            "Note: Samurai requires a 'prompt' parameter (e.g., detection results from YOLO)."
        )

    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert to base-level dict (Level 2).

        To be implemented when Samurai predictor is activated.

        Raises:
            NotImplementedError: Samurai support is not yet implemented
        """
        raise NotImplementedError(
            "Samurai support is planned for future release. "
            "The Samurai predictor is currently commented out in the codebase."
        )

    def to_human(self) -> Dict[str, Any]:
        """
        Rich dict with all information.

        To be implemented when Samurai predictor is activated.

        Raises:
            NotImplementedError: Samurai support is not yet implemented
        """
        raise NotImplementedError(
            "Samurai support is planned for future release. "
            "The Samurai predictor is currently commented out in the codebase."
        )

    @property
    def num_predictions(self) -> int:
        """Number of tracked objects across frames."""
        return len(self.track_ids) if self.track_ids else 0

    def visualize(self, **kwargs):
        """
        Visualize Samurai tracking results.

        To be implemented when Samurai predictor is activated.

        Raises:
            NotImplementedError: Samurai support is not yet implemented
        """
        raise NotImplementedError(
            "Samurai support is planned for future release. "
            "The Samurai predictor is currently commented out in the codebase."
        )
