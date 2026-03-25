"""Core prediction interface for dependency inversion.

This module defines the abstract interface for all predictors, supporting
a three-level return system:
- Level 1 (raw): Framework-native output (fastest, default)
- Level 2 (base): Portable dict for MLflow/pipelines
- Level 3 (rich): Full result object with methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, Protocol, TypeVar

from bovi_core.config import Config

# Type variable for input data type
# - np.ndarray | list[np.ndarray] | list[str] for image predictors
# - dict[str, object] | list[dict[str, object]] for tabular predictors
InputT = TypeVar("InputT")

# Type variable for the rich result type
# - YoloPredictionResult for YOLO
# - LactationPredictionResult for lactation
ResultT = TypeVar("ResultT")

# Type variable for the model instance type
# - YOLO for ultralytics
# - LactationAutoencoderModel for lactation
ModelT = TypeVar("ModelT")


class CallableModel(Protocol):
    """Protocol for models that can be called for inference.

    This provides structural typing for any model that implements __call__.
    Use this when you don't want to import the specific model class to avoid
    circular imports.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class PredictionInterface(ABC, Generic[InputT, ResultT, ModelT]):
    """Abstract interface that all predictors must implement.

    This interface enables dependency inversion, allowing models to depend on
    abstractions rather than concrete predictor implementations.

    Generic Parameters:
        InputT: The type of input data this predictor accepts
        ResultT: The type of rich result this predictor returns

    Subclasses should set the `result_class` attribute to specify which
    prediction result class they use (e.g., YoloPredictionResult).

    Example:
        class YOLOPredictor(PredictionInterface[
            np.ndarray | list[np.ndarray] | list[str],
            YoloPredictionResult,
            YOLO
        ]):
            result_class = YoloPredictionResult
            ...

        class LactationPredictor(PredictionInterface[
            dict[str, object] | list[dict[str, object]],
            LactationPredictionResult,
            LactationAutoencoderModel
        ]):
            result_class = LactationPredictionResult
            ...
    """

    # Subclasses should override this to specify their result class
    result_class: type[ResultT] | None = None

    def __init__(self, config: Config) -> None:
        """Initialize predictor with configuration.

        Args:
            config: Application configuration instance
        """
        self.config = config
        self.model_instance: ModelT | None = None
        self.initialize()

    @abstractmethod
    def initialize(self) -> None:
        """Initialize predictor resources and configurations."""
        pass

    def set_model_instance(self, model_instance: ModelT) -> None:
        """Set the loaded model instance after model initialization.

        Args:
            model_instance: The loaded model (torch.nn.Module, YOLO, TF model, etc.)
        """
        self.model_instance = model_instance

    @abstractmethod
    def predict(
        self,
        data: InputT,
        return_format: Literal["raw", "base", "rich"] = "raw",
        prompt: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Perform prediction on input data.

        Args:
            data: Input data (type depends on predictor implementation)
            return_format: Output format (default: "raw")
                - "raw": Framework-native output (fastest, no conversion)
                - "base": Portable dict for MLflow signatures and pipelines
                - "rich": Full result object with methods (visualization, filtering)
            prompt: Optional prompts for models that support them (SAM, Samurai)
            **kwargs: Additional model-specific parameters

        Returns:
            Prediction in the requested format:
            - "raw": Framework-specific output (e.g., ultralytics.Results, tf.Tensor)
            - "base": Serializable dict with metadata
            - "rich": ResultT instance with methods

        Raises:
            PredictionError: If prediction fails

        Example:
            >>> # Level 1: Get raw output (fastest)
            >>> raw = predictor.predict(image, return_format="raw")

            >>> # Level 2: Get base dict for MLflow
            >>> base_dict = predictor.predict(image, return_format="base")
            >>> signature = infer_signature(input, base_dict)

            >>> # Level 3: Get rich object for visualization
            >>> result = predictor.predict(image, return_format="rich")
            >>> result.draw_on_image()
        """
        pass

    def cleanup(self) -> None:
        """Clean up resources. Default implementation does nothing."""
        pass

    def __enter__(self) -> PredictionInterface[InputT, ResultT, ModelT]:
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()
