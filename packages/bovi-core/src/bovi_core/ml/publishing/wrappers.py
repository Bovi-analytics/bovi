"""Generic MLflow pyfunc wrappers for exported framework artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import mlflow.pyfunc


class PyTorchModelWrapper(mlflow.pyfunc.PythonModel):  # type: ignore[reportPrivateImportUsage]
    """Load and serve a complete PyTorch module artifact."""

    def load_context(self, context: Any) -> None:
        import torch

        self.model = torch.load(context.artifacts["model_path"], weights_only=False)

    def predict(self, context: Any, model_input: Any, params: dict[str, Any] | None = None) -> Any:
        import torch

        if hasattr(model_input, "values"):
            model_input = model_input.values
        if isinstance(model_input, dict):
            input_tensor = {
                key: torch.as_tensor(value, dtype=torch.float32)
                for key, value in model_input.items()
            }
        else:
            input_tensor = torch.as_tensor(model_input, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        if isinstance(output, dict):
            return {key: value.cpu().numpy() for key, value in output.items()}
        return output.cpu().numpy()


class KerasModelWrapper(mlflow.pyfunc.PythonModel):  # type: ignore[reportPrivateImportUsage]
    """Load and serve a complete Keras model artifact."""

    def load_context(self, context: Any) -> None:
        import tensorflow as tf

        self.model = tf.keras.models.load_model(context.artifacts["model_path"])

    def predict(self, context: Any, model_input: Any, params: dict[str, Any] | None = None) -> Any:
        import numpy as np

        if hasattr(model_input, "values"):
            input_data = model_input.values.astype(np.float32)
        elif isinstance(model_input, dict):
            input_data = {
                key: np.asarray(value, dtype=np.float32) for key, value in model_input.items()
            }
        else:
            input_data = np.asarray(model_input, dtype=np.float32)
        return self.model.predict(input_data)


class TensorFlowSavedModelWrapper(mlflow.pyfunc.PythonModel, ABC):  # type: ignore[reportPrivateImportUsage]
    """Serve a TensorFlow SavedModel with model-specific semantic input names."""

    def load_context(self, context: Any) -> None:
        import tensorflow as tf

        self.model = tf.saved_model.load(context.artifacts["model_path"])
        self._signature = self.model.signatures["serving_default"]

    @abstractmethod
    def get_input_name_mapping(self) -> dict[str, str]:
        """Map semantic input names to SavedModel signature names."""
        raise NotImplementedError

    def predict(
        self,
        context: Any,
        model_input: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        import tensorflow as tf

        mapping = self.get_input_name_mapping()
        results: list[dict[str, Any]] = []
        for instance in model_input:
            inputs = {
                mapping.get(key, key): tf.constant(value, dtype=tf.float32)
                for key, value in instance.items()
            }
            output = self._signature(**inputs)
            if isinstance(output, dict):
                results.append({key: value.numpy() for key, value in output.items()})
            else:
                results.append({"output": output.numpy()})
        return results
