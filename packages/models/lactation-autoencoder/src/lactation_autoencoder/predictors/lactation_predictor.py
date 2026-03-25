"""
Lactation Autoencoder Predictor.

This module provides the prediction interface for the lactation autoencoder model.
It handles preprocessing of input data and returns predictions in multiple formats.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from bovi_core.ml import PredictionInterface, PredictorRegistry
from bovi_core.ml.predictors.prediction_interface import CallableModel
from typing_extensions import override

from lactation_autoencoder.predictors.results.lactation_prediction_result import (
    LactationPredictionResult,
)


class PredictionError(Exception):
    """Base exception for prediction errors."""

    model_name: str
    original_exception: Exception | None

    def __init__(
        self, message: str, model_name: str, original_exception: Exception | None = None
    ) -> None:
        super().__init__(message)
        self.model_name = model_name
        self.original_exception = original_exception


# Type alias for lactation input data
LactationInput = dict[str, object] | list[dict[str, object]]


@PredictorRegistry.register("autoencoder")
class LactationPredictor(
    PredictionInterface[LactationInput, LactationPredictionResult, CallableModel]
):
    """
    Lactation predictor implementing the standard prediction interface.

    Supports three-level prediction returns:
    - Level 1 (raw): TensorFlow tensor output (fastest)
    - Level 2 (base): Portable dict for MLflow signatures
    - Level 3 (rich): LactationPredictionResult with methods
    """

    # Declare the result class this predictor uses
    result_class = LactationPredictionResult

    @override
    def initialize(self) -> None:
        """Initialize lactation predictor."""
        # Model instance will be set via set_model_instance()
        pass

    @override
    def set_model_instance(self, model_instance: CallableModel) -> None:
        """
        Set the model instance.

        The model_instance should be the LactationAutoencoderModel which implements
        __call__() for consistent inference across all model types.
        """
        super().set_model_instance(model_instance)

    @override
    def predict(
        self,
        data: dict[str, object] | list[dict[str, object]],
        return_format: Literal["raw", "base", "rich"] = "raw",
        prompt: dict[str, object] | None = None,
        **kwargs: object,
    ) -> object:
        """
        Perform lactation prediction with three-level return options.

        Args:
            data: Input data dict(s) with keys:
                - 'milk': Milk production sequence (304 days, normalized 0-1)
                - 'events': Event indices (304 days)
                - 'parity': Parity number
                - 'herd_stats': Herd statistics (optional)
            return_format: Output format
                - "raw": TensorFlow tensor (fastest, default)
                - "base": Portable dict for MLflow
                - "rich": LactationPredictionResult with methods
            prompt: Unused for lactation model
            **kwargs: Additional parameters for model

        Returns:
            Prediction in requested format

        Raises:
            PredictionError: If prediction fails

        Example:
            >>> # Get raw output (fastest)
            >>> raw = predictor.predict(input_dict, return_format="raw")
            >>>
            >>> # Get base dict for MLflow signatures
            >>> base_dict = predictor.predict(input_dict, return_format="base")
            >>>
            >>> # Get rich object with plotting methods
            >>> result = predictor.predict(input_dict, return_format="rich")
            >>> result.plot_prediction()
        """
        if self.model_instance is None:
            raise PredictionError("Model instance not set", "lactation_autoencoder")

        # Handle single dict or list of dicts
        is_batch = isinstance(data, list)
        data_list: list[dict[str, object]] = data if is_batch else [data]

        try:
            # Prepare model inputs based on what the model expects
            model_inputs = self._prepare_inputs(data_list)

            # Get raw predictions from model (Level 1)
            # Use consistent __call__ interface across all model types
            raw_predictions = self.model_instance(**model_inputs)

            # Return based on format request
            if return_format == "raw":
                return raw_predictions

            # Convert to rich result object (Level 3)
            first_item = data_list[0]
            first_item_metadata = first_item.get("metadata")
            metadata_dict: dict[str, object] = (
                dict(first_item_metadata) if isinstance(first_item_metadata, dict) else {}
            )
            rich_result = LactationPredictionResult.from_raw(
                raw_predictions,
                input_data=first_item if not is_batch else None,
                metadata=metadata_dict,
            )

            if return_format == "rich":
                return rich_result

            if return_format == "base":
                return rich_result.to_serializable()

            # Fallback to raw
            return raw_predictions

        except Exception as e:
            raise PredictionError(
                f"Lactation prediction failed: {e!s}", "lactation_autoencoder", e
            ) from e

    def _prepare_inputs(self, data: list[dict[str, object]]) -> dict[str, tf.Tensor]:
        """
        Prepare inputs for the TFSMLayer model.

        The model expects:
        - input_11: milk (batch_size, 304, 1)
        - input_13: events (batch_size, 304)
        - input_12: parity (batch_size, 1)
        - input_15: herd_stats (batch_size, 10)

        Args:
            data: List of input dicts

        Returns:
            Dict with tensor inputs in correct format
        """
        # Prepare tensors in the order expected by model signature
        # Based on inspection: input_11 (milk), input_12 (parity),
        # input_13 (events), input_15 (herd_stats)

        # Milk input (batch_size, 304, 1) - input_11
        milk_arrays = [np.asarray(d["milk"], dtype=np.float32) for d in data]
        milk_batch: npt.NDArray[np.float32] = np.stack(milk_arrays)
        if len(milk_batch.shape) == 2:
            milk_batch = np.expand_dims(milk_batch, axis=-1)
        input_11 = tf.convert_to_tensor(milk_batch, dtype=tf.float32)

        # Parity input (batch_size, 1) - input_12
        parity_list: list[list[float]] = []
        for d in data:
            p = d["parity"]
            if isinstance(p, (int, float)):
                parity_list.append([float(p)])
            elif isinstance(p, np.ndarray):
                if p.shape and p.shape[0] == 1:
                    parity_list.append([float(p[0])])
                else:
                    parity_list.append([float(p.flat[0])])
            else:
                # Convert via string for any other types (numpy scalars, etc.)
                parity_list.append([float(str(p))])
        parity_batch: npt.NDArray[np.float32] = np.array(parity_list, dtype=np.float32)
        input_12 = tf.convert_to_tensor(parity_batch, dtype=tf.float32)

        # Event input (batch_size, 304) - input_13
        events_arrays = [np.asarray(d["events"], dtype=np.float32) for d in data]
        events_batch: npt.NDArray[np.float32] = np.stack(events_arrays)
        input_13 = tf.convert_to_tensor(events_batch, dtype=tf.float32)

        # Herd stats input (batch_size, num_stats) - input_15
        herd_stats_arrays = [np.asarray(d["herd_stats"], dtype=np.float32) for d in data]
        herd_stats_batch: npt.NDArray[np.float32] = np.stack(herd_stats_arrays)
        input_15 = tf.convert_to_tensor(herd_stats_batch, dtype=tf.float32)

        return {
            "input_11": input_11,
            "input_12": input_12,
            "input_13": input_13,
            "input_15": input_15,
        }
