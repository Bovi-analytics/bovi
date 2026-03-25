"""YOLO predictor implementing the standard prediction interface.

Supports three-level prediction returns:
- Level 1 (raw): ultralytics.Results object (fastest, framework-specific)
- Level 2 (base): Portable dict for MLflow signatures and pipelines
- Level 3 (rich): YoloPredictionResult with methods (visualization, filtering)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from bovi_core.ml import PredictionInterface, PredictorRegistry
from bovi_core.ml.predictors.prediction_interface import CallableModel
from typing_extensions import override

from bovi_yolo.predictors.results.yolo_prediction_result import (
    YoloPredictionResult,
)

# Type alias for YOLO input data
YOLOInput = npt.NDArray[np.uint8] | list[npt.NDArray[np.uint8]] | list[str]


class PredictionError(Exception):
    """Base exception for prediction errors."""

    model_name: str
    original_exception: Exception | None

    def __init__(
        self,
        message: str,
        model_name: str,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.model_name = model_name
        self.original_exception = original_exception


@PredictorRegistry.register("yolo")
class YOLOPredictor(PredictionInterface[YOLOInput, YoloPredictionResult, CallableModel]):
    """YOLO predictor implementing the standard prediction interface.

    Supports three-level prediction returns:
    - Level 1 (raw): ultralytics.Results object (fastest, framework-specific)
    - Level 2 (base): Portable dict for MLflow signatures and pipelines
    - Level 3 (rich): YoloPredictionResult with methods (visualization, filtering)
    """

    # Declare the result class this predictor uses
    result_class = YoloPredictionResult

    @override
    def initialize(self) -> None:
        """Initialize YOLO predictor."""
        # Model instance will be set via set_model_instance()

    @override
    def set_model_instance(self, model_instance: CallableModel) -> None:
        """Set the YOLO model instance.

        The model_instance should be the YOLOModel which implements
        __call__() for consistent inference across all model types.

        Args:
            model_instance: YOLOModel instance (implements __call__).
        """
        super().set_model_instance(model_instance)

    @override
    def predict(
        self,
        data: npt.NDArray[np.uint8] | list[npt.NDArray[np.uint8]] | list[str],
        return_format: Literal["raw", "base", "rich"] = "raw",
        prompt: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any | dict[str, Any] | YoloPredictionResult:
        """Perform YOLO prediction with three-level return options.

        Args:
            data: Input images (array, list of arrays, or file paths).
            return_format: Output format
                - "raw": ultralytics.Results object (fastest, default)
                - "base": Portable dict for MLflow and pipelines
                - "rich": YoloPredictionResult with methods
            prompt: Optional prompts (unused for YOLO).
            **kwargs: Additional YOLO parameters (conf, iou, imgsz, etc.).

        Returns:
            Prediction in requested format.

        Raises:
            PredictionError: If prediction fails.

        Example:
            >>> raw = predictor.predict(image, return_format="raw")
            >>> base_dict = predictor.predict(image, return_format="base")
            >>> result = predictor.predict(image, return_format="rich")
            >>> result.draw_on_image()
        """
        if self.model_instance is None:
            raise PredictionError("Model instance not set", "yolo")

        try:
            # Get raw YOLO output (Level 1)
            raw_results = self.model_instance(data, **kwargs)

            if return_format == "raw":
                return raw_results

            # Convert to rich result object (Level 3)
            original_image = data if isinstance(data, np.ndarray) else None

            rich_result = YoloPredictionResult.from_raw(
                raw_results,
                original_image=original_image,
                class_names_map=getattr(self.model_instance, "names", None),
            )

            if return_format == "rich":
                return rich_result

            if return_format == "base":
                return rich_result.to_serializable()

            # Fallback to raw
            return raw_results

        except Exception as e:
            raise PredictionError(f"YOLO prediction failed: {e!s}", "yolo", e) from e
