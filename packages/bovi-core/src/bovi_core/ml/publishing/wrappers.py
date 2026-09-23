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
    """Serve named, batched arrays using a SavedModel's shapes and dtypes.

    Subclasses map public input names to native signature names. Model-specific
    preprocessing belongs in the subclass, not in this framework wrapper.
    Supply an explicit MLflow tensor signature when saving or publishing.
    """

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
        model_input,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # MLflow validates named tensor batches through the saved signature,
        # not its list-of-records type-hint inference.
        import numpy as np
        import tensorflow as tf

        mapping = self.get_input_name_mapping()
        specs = self._signature.structured_input_signature[1]
        if set(mapping.values()) != set(specs) or len(mapping) != len(specs):
            raise ValueError("Input name mapping does not match the SavedModel signature")
        if not isinstance(model_input, dict) or set(model_input) != set(mapping):
            raise ValueError(f"Expected a dictionary of batched arrays with keys {sorted(mapping)}")

        inputs = {}
        batch_size = None
        for public_name, native_name in mapping.items():
            spec = specs[native_name]
            array = np.asarray(model_input[public_name], dtype=spec.dtype.as_numpy_dtype)
            if array.ndim == 0 or array.shape[0] == 0:
                raise ValueError(f"{public_name} must have a nonempty batch dimension")
            if batch_size is not None and array.shape[0] != batch_size:
                raise ValueError("All inputs must have the same batch size")
            batch_size = array.shape[0]
            if not spec.shape.is_compatible_with(array.shape):
                raise ValueError(f"{public_name} has shape {array.shape}; expected {spec.shape}")
            inputs[native_name] = tf.convert_to_tensor(array, dtype=spec.dtype)

        output = self._signature(**inputs)
        return {key: value.numpy() for key, value in output.items()}
