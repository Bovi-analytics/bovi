"""Pyfunc wrapper for Keras models.

This wrapper bundles Keras models with config files as model artifacts,
ensuring config survives MLflow run deletion.

Usage in unity_catalog.py:
    mlflow.pyfunc.log_model(
        python_model=KerasModelWrapper(),
        artifacts={
            "model_path": path_to_saved_model,
            "config_yaml": path_to_config,
            "pyproject_toml": path_to_pyproject,
        },
        ...
    )
"""

from typing import Any, Dict, Optional

import mlflow.pyfunc


class KerasModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Pyfunc wrapper for Keras models.

    Stores Keras model with config files as bundled artifacts.
    This ensures config files survive MLflow run deletion since they're
    stored with the model artifacts, not as run artifacts.
    """

    def load_context(self, context):
        """Load Keras model from artifacts."""
        import tensorflow as tf

        model_path = context.artifacts["model_path"]
        self.model = tf.keras.models.load_model(model_path)

    def predict(
        self, context: Any, model_input: Any, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Run inference (required by MLflow, but typically not used in practice).

        In practice, predictions go through load_from_unity_catalog() which
        reconstructs the full Model + Predictor with proper pre/post processing.

        This method is only used during log_model() validation.

        Args:
            context: MLflow context (provided by framework)
            model_input: Input data (pandas DataFrame, numpy array, or dict)
            params: Optional parameters (unused)

        Returns:
            Model predictions as numpy array
        """
        import numpy as np

        # Handle different input types
        if hasattr(model_input, "values"):
            # pandas DataFrame
            input_data = model_input.values.astype(np.float32)
        elif isinstance(model_input, dict):
            # Dict input - pass directly to model (for multi-input models)
            input_data = {k: np.array(v, dtype=np.float32) for k, v in model_input.items()}
        else:
            # Assume numpy array
            input_data = np.array(model_input, dtype=np.float32)

        # Run inference
        output = self.model.predict(input_data)
        return output
