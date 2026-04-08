"""Pyfunc wrapper for PyTorch models.

This wrapper bundles PyTorch models with config files as model artifacts,
ensuring config survives MLflow run deletion.

Usage in unity_catalog.py:
    mlflow.pyfunc.log_model(
        python_model=PyTorchModelWrapper(),
        artifacts={
            "model_path": path_to_model,
            "config_yaml": path_to_config,
            "pyproject_toml": path_to_pyproject,
        },
        ...
    )
"""

from typing import Any, Dict, Optional

import mlflow.pyfunc


class PyTorchModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Pyfunc wrapper for PyTorch models.

    Stores PyTorch model with config files as bundled artifacts.
    This ensures config files survive MLflow run deletion since they're
    stored with the model artifacts, not as run artifacts.
    """

    def load_context(self, context):
        """Load PyTorch model from artifacts."""
        import torch

        model_path = context.artifacts["model_path"]
        self.model = torch.load(model_path, weights_only=False)

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
            Model predictions
        """
        import torch

        # Handle different input types
        if hasattr(model_input, "values"):
            # pandas DataFrame
            input_tensor = torch.tensor(model_input.values, dtype=torch.float32)
        elif isinstance(model_input, dict):
            # Dict input - convert each value to tensor
            input_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in model_input.items()}
        else:
            # Assume numpy array or already tensor
            input_tensor = torch.tensor(model_input, dtype=torch.float32)

        # Set model to eval mode and run inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)

        # Convert output to numpy
        if isinstance(output, dict):
            return {k: v.cpu().numpy() for k, v in output.items()}
        return output.cpu().numpy()
