"""Unity Catalog utilities for lactation model."""

from typing import Dict

from bovi_core.ml.models.tensorflow_savedmodel_wrapper import (
    TensorFlowSavedModelWrapper,
)


class LactationSavedModelWrapper(TensorFlowSavedModelWrapper):
    """
    Pyfunc wrapper for lactation autoencoder SavedModel.

    Maps semantic names (milk, events, parity, herd_stats) to
    TensorFlow generic names (input_11, input_12, input_13, input_15).
    """

    def get_input_name_mapping(self) -> Dict[str, str]:
        """Return lactation-specific input name mapping."""
        return {
            "milk": "input_11",  # (batch, 304, 1) milk recordings
            "parity": "input_12",  # (batch, 1) parity number
            "events": "input_13",  # (batch, 304) event flags
            "herd_stats": "input_15",  # (batch, 10) herd statistics
        }
