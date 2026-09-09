"""MLflow serving for the lactation autoencoder's normalized feature batches."""

from typing import Any

import numpy as np
from bovi_core.ml.publishing.wrappers import TensorFlowSavedModelWrapper


class LactationSavedModelWrapper(TensorFlowSavedModelWrapper):
    """Accept batched milk, events, parity and herd statistics arrays.

    All public inputs are two-dimensional, including parity shaped (batch, 1).
    Milk gains the channel dimension required by the native SavedModel. Outputs
    are named, normalized native arrays, not rich predictor result objects.
    """

    def get_input_name_mapping(self) -> dict[str, str]:
        return {
            "milk": "input_11",
            "parity": "input_12",
            "events": "input_13",
            "herd_stats": "input_15",
        }

    def predict(self, context: Any, model_input, params: dict[str, Any] | None = None):
        if not isinstance(model_input, dict):
            raise ValueError("Expected a dictionary of batched lactation arrays")
        arrays = {key: np.asarray(value) for key, value in model_input.items()}
        if set(arrays) != set(self.get_input_name_mapping()):
            raise ValueError("Expected milk, events, parity and herd_stats")
        for key, array in arrays.items():
            if array.ndim != 2 or not np.isfinite(array).all():
                raise ValueError(f"{key} must be a finite two-dimensional array")
        arrays["milk"] = arrays["milk"][..., None]
        return super().predict(context, arrays, params)
