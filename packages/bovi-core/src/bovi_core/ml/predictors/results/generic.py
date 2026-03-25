"""Generic prediction result for simple models.

Used as fallback for sklearn, keras, regression models, and other frameworks
that return simple arrays or dictionaries.
"""

from typing import Any, Dict, Optional, Union

import numpy as np

from .base import BasePredictionResult


class GenericPredictionResult(BasePredictionResult):
    """
    Generic prediction result for sklearn, keras, simple arrays.

    This is a minimal wrapper for models that don't need rich visualization
    or manipulation methods. Simply wraps predictions for serialization.

    Examples:
        - Sklearn classifiers/regressors (returns np.ndarray)
        - Keras models (returns np.ndarray or dict)
        - Simple regression models
        - Any model with simple array outputs
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Dict[str, np.ndarray]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize generic prediction result.

        Args:
            predictions: Predictions as numpy array or dict of arrays
            metadata: Optional metadata
        """
        super().__init__(metadata)
        self.predictions = predictions

    @classmethod
    def from_raw(cls, raw_output: Union[np.ndarray, Dict], **kwargs):
        """
        Create from numpy array or dict of arrays (Level 1 → Level 3).

        Args:
            raw_output: Raw model output (np.ndarray or dict)
            **kwargs: Additional context (will be added to metadata)

        Returns:
            GenericPredictionResult instance

        Example:
            >>> sklearn_output = model.predict(X)  # np.ndarray
            >>> result = GenericPredictionResult.from_raw(sklearn_output)
        """
        metadata = kwargs.get("metadata", {})
        # Merge any extra kwargs into metadata
        for key, value in kwargs.items():
            if key != "metadata":
                metadata[key] = value

        return cls(predictions=raw_output, metadata=metadata)

    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert to base-level dict (Level 2).

        Handles both array and dict outputs, converting to JSON-serializable format.

        Returns:
            Serializable dict with predictions and metadata

        Example:
            >>> # Array output
            >>> result.to_serializable()
            {
                "predictions": [[0.1, 0.9], [0.8, 0.2]],
                "num_predictions": 2,
                "metadata": {
                    "model_type": "generic",
                    "prediction_shape": [2, 2],
                    "prediction_dtype": "float64"
                }
            }

            >>> # Dict output
            >>> result.to_serializable()
            {
                "class_probs": [[0.1, 0.9]],
                "embeddings": [[0.5, 0.3, ...]],
                "num_predictions": 1,
                "metadata": {
                    "model_type": "generic",
                    "output_keys": ["class_probs", "embeddings"]
                }
            }
        """
        if isinstance(self.predictions, np.ndarray):
            # Simple array output
            pred_dict = {"predictions": self.predictions.tolist()}
            metadata = {
                **self.metadata,
                "model_type": "generic",
                "prediction_shape": list(self.predictions.shape),
                "prediction_dtype": str(self.predictions.dtype),
            }
        elif isinstance(self.predictions, dict):
            # Dict output (multiple outputs)
            pred_dict = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.predictions.items()
            }
            metadata = {
                **self.metadata,
                "model_type": "generic",
                "output_keys": list(self.predictions.keys()),
            }
        else:
            # Fallback for other types
            pred_dict = {"predictions": self.predictions}
            metadata = {**self.metadata, "model_type": "generic"}

        return {
            **pred_dict,
            "num_predictions": self.num_predictions,
            "metadata": metadata,
        }

    @property
    def num_predictions(self) -> int:
        """
        Number of predictions.

        Returns:
            Count of predictions (batch size)
        """
        if isinstance(self.predictions, np.ndarray):
            return len(self.predictions) if self.predictions.ndim > 0 else 1
        elif isinstance(self.predictions, dict):
            # Use first key to determine count
            first_key = next(iter(self.predictions.keys()))
            first_value = self.predictions[first_key]
            if isinstance(first_value, np.ndarray):
                return len(first_value) if first_value.ndim > 0 else 1
        return 1
