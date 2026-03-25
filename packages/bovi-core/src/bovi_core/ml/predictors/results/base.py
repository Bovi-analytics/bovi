"""Base classes for prediction results.

This module defines the abstract base classes for all prediction results,
establishing a three-level return system:
- Level 1 (raw): Framework-native output (fastest, no conversion)
- Level 2 (base): Portable dict for MLflow/pipelines (serializable)
- Level 3 (rich): Full object with methods (visualization, filtering)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BasePredictionResult(ABC):
    """
    Abstract base for all prediction results.

    This class defines the minimal contract for serialization and metadata.
    All prediction results must be able to convert to a portable dictionary
    format (Level 2) that can be used for MLflow signatures and model chaining.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize base prediction result.

        Args:
            metadata: Optional metadata dict. Will be included in serialization.
        """
        self.metadata = metadata or {}

    @classmethod
    @abstractmethod
    def from_raw(cls, raw_output: Any, **kwargs) -> "BasePredictionResult":
        """
        Construct prediction result from framework's raw output.

        This factory method converts Level 1 (raw) outputs to Level 3 (rich).

        Args:
            raw_output: Framework-specific output (e.g., ultralytics.Results,
                       torch.Tensor, np.ndarray, dict)
            **kwargs: Additional context (e.g., original_image, class_names_map)

        Returns:
            Concrete prediction result instance

        Example:
            >>> yolo_results = model(image)  # Level 1: raw
            >>> result = YoloPredictionResult.from_raw(yolo_results)  # Level 3
        """
        pass

    @abstractmethod
    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert to base-level dict (Level 2).

        This produces a portable, MLflow-compatible dictionary with:
        - Original format preserved (e.g., boxes in native format)
        - Format metadata (box_format, color_format, etc.)
        - No large tensors or images (lightweight)
        - Model-specific schema (YOLO has boxes, SAM has masks, etc.)

        All implementations must include:
        - num_predictions: int
        - metadata: dict with format information

        Returns:
            Serializable dict ready for MLflow or downstream models

        Example:
            >>> result.to_serializable()
            {
                "boxes_xyxy": [[10, 20, 100, 200]],
                "scores": [0.95],
                "num_predictions": 1,
                "metadata": {
                    "model_type": "yolo",
                    "box_format": "xyxy",
                    "color_format": "bgr"
                }
            }
        """
        pass

    @property
    @abstractmethod
    def num_predictions(self) -> int:
        """
        Number of predictions in this result.

        Returns:
            Count of predictions (detections, instances, samples, etc.)
        """
        pass


class HumanReadablePredictionResult(BasePredictionResult):
    """
    Extended base for results with rich visualization and manipulation.

    This class adds methods for human interaction: visualization, filtering,
    format conversion, etc. Used for computer vision models (YOLO, SAM, Samurai)
    where rich manipulation is needed.
    """

    @abstractmethod
    def to_human(self) -> Dict[str, Any]:
        """
        Rich dict with all information and computed fields.

        This produces a comprehensive dictionary with:
        - All data from to_serializable()
        - Multiple format variations (e.g., boxes in xyxy, xywh, ltwh)
        - Additional computed fields
        - Rich metadata

        Used for debugging, logging, or preparing data for visualization.

        Returns:
            Rich dict with all available information

        Example:
            >>> result.to_human()
            {
                "boxes_xyxy": [[10, 20, 100, 200]],
                "boxes_xywh": [[55, 110, 90, 180]],  # Additional format
                "boxes_ltwh": [[10, 20, 90, 180]],   # Additional format
                "scores": [0.95],
                "num_predictions": 1,
                "has_masks": False,
                "metadata": {...}
            }
        """
        pass

    @abstractmethod
    def visualize(self, **kwargs):
        """
        Visualize predictions interactively.

        Implementation depends on model type:
        - YOLO: Draw boxes/masks on image
        - SAM: Show segmentation masks
        - etc.

        Args:
            **kwargs: Visualization options (figsize, colors, etc.)
        """
        pass
