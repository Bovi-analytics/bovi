"""SAM (Segment Anything Model) prediction result.

Placeholder for future SAM integration.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base import HumanReadablePredictionResult


class SamPredictionResult(HumanReadablePredictionResult):
    """
    SAM (Segment Anything Model) prediction result.

    Placeholder for future implementation when SAM predictor is activated.
    Will handle segmentation masks, scores, and optional bounding boxes.
    """

    def __init__(
        self,
        masks: Optional[List[np.ndarray]] = None,
        scores: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
        original_image: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize SAM prediction result.

        Args:
            masks: Binary segmentation masks
            scores: Confidence scores for each mask
            boxes: Optional bounding boxes (often from prompt)
            logits: Optional mask logits from the model
            original_image: Original input image
            metadata: Additional metadata
        """
        super().__init__(metadata)
        self.masks = masks
        self.scores = scores
        self.boxes = boxes
        self.logits = logits
        self.original_image = original_image

    @classmethod
    def from_raw(cls, raw_output: Any, **kwargs):
        """
        Create from SAM output (to be implemented when SAM predictor is activated).

        Args:
            raw_output: SAM model output
            **kwargs: Additional context

        Returns:
            SamPredictionResult instance

        Raises:
            NotImplementedError: SAM support is not yet implemented
        """
        raise NotImplementedError(
            "SAM support is planned for future release. "
            "The SAM predictor is currently commented out in the codebase."
        )

    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert to base-level dict (Level 2).

        To be implemented when SAM predictor is activated.

        Raises:
            NotImplementedError: SAM support is not yet implemented
        """
        raise NotImplementedError(
            "SAM support is planned for future release. "
            "The SAM predictor is currently commented out in the codebase."
        )

    def to_human(self) -> Dict[str, Any]:
        """
        Rich dict with all information.

        To be implemented when SAM predictor is activated.

        Raises:
            NotImplementedError: SAM support is not yet implemented
        """
        raise NotImplementedError(
            "SAM support is planned for future release. "
            "The SAM predictor is currently commented out in the codebase."
        )

    @property
    def num_predictions(self) -> int:
        """Number of segmentation masks."""
        return len(self.masks) if self.masks else 0

    def visualize(self, **kwargs):
        """
        Visualize SAM masks.

        To be implemented when SAM predictor is activated.

        Raises:
            NotImplementedError: SAM support is not yet implemented
        """
        raise NotImplementedError(
            "SAM support is planned for future release. "
            "The SAM predictor is currently commented out in the codebase."
        )
