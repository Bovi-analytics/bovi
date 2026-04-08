"""Abstract pyfunc wrapper for TensorFlow SavedModels.

This wrapper is used to save pure TensorFlow SavedModels to Unity Catalog.
It's only used for storage/retrieval - actual predictions go through the
Model + Predictor classes after loading via load_from_unity_catalog().

Subclasses must implement get_input_name_mapping() to provide model-specific
name mappings from semantic names to TensorFlow generic input names.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import mlflow.pyfunc
import tensorflow as tf


class TensorFlowSavedModelWrapper(mlflow.pyfunc.PythonModel, ABC):
    """
    Abstract base wrapper for TensorFlow SavedModels.

    Subclasses must implement get_input_name_mapping() to map semantic names
    to TensorFlow generic input names.
    """

    def load_context(self, context):
        """Load SavedModel from artifacts."""
        self.model = tf.saved_model.load(context.artifacts["model_path"])
        self._signature = self.model.signatures["serving_default"]
        # Store artifacts dict for access after loading (e.g., for config files)
        self._artifacts = context.artifacts

    @abstractmethod
    def get_input_name_mapping(self) -> Dict[str, str]:
        """
        Return mapping from semantic names to TensorFlow generic names.

        Returns:
            Dict mapping semantic names (e.g., 'milk') to TF names (e.g., 'input_11')

        Example:
            >>> def get_input_name_mapping(self):
            ...     return {
            ...         "milk": "input_11",
            ...         "parity": "input_12",
            ...         "events": "input_13",
            ...         "herd_stats": "input_15",
            ...     }
        """
        pass

    def predict(
        self, context: Any, model_input: Dict[str, Any], params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run inference (required by MLflow, but not used in practice).

        In practice, predictions go through load_from_unity_catalog() which
        reconstructs the full Model + Predictor with three-level returns.

        This method is only used during log_model() validation. It uses
        get_input_name_mapping() from the subclass to map semantic names
        to TensorFlow generic names.

        Args:
            context: MLflow context (provided by framework)
            model_input: Dict with semantic input names
            params: Optional parameters (unused)

        Returns:
            Dict with model outputs as numpy arrays
        """
        # Get name mapping from subclass
        name_mapping = self.get_input_name_mapping()

        # Convert dict inputs to TensorFlow tensors with mapped names
        # Explicitly cast to float32 to match TensorFlow signature expectations
        tf_inputs = {
            name_mapping.get(k, k): tf.constant(v, dtype=tf.float32) for k, v in model_input.items()
        }

        # Run inference
        output = self._signature(**tf_inputs)

        # Convert to numpy
        if isinstance(output, dict):
            return {k: v.numpy() for k, v in output.items()}
        return output.numpy()
