"""
Lactation Prediction Result.

This module provides the result object for lactation predictions with
visualization and analysis methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import tensorflow as tf

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class LactationPredictionResult:
    """Result object for lactation predictions with visualization and analysis methods."""

    predictions: npt.NDArray[np.float32]
    input_data: dict[str, object] | None
    metadata: dict[str, object]

    def __init__(
        self,
        predictions: npt.NDArray[np.float32],
        input_data: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """
        Initialize lactation prediction result.

        Args:
            predictions: Predicted milk production sequence (304 days)
            input_data: Original input data dict
            metadata: Additional metadata (animal_id, herd_id, etc.)
        """
        self.predictions = predictions
        self.input_data = input_data
        self.metadata = metadata or {}

    @classmethod
    def from_raw(
        cls,
        raw_predictions: object,
        input_data: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> LactationPredictionResult:
        """
        Create result from raw TensorFlow output.

        Args:
            raw_predictions: Raw output from TFSMLayer
            input_data: Original input data
            metadata: Metadata dict

        Returns:
            LactationPredictionResult instance
        """
        # Convert TensorFlow tensor to numpy if needed
        predictions: object
        if isinstance(raw_predictions, dict):
            # Extract the output tensor from dict (TFSMLayer returns dict)
            predictions = next(iter(raw_predictions.values()))
        else:
            predictions = raw_predictions

        if tf.is_tensor(predictions):
            predictions = cast(tf.Tensor, predictions).numpy()

        # Ensure predictions are numpy array
        predictions_array: npt.NDArray[np.float32] = np.asarray(predictions, dtype=np.float32)

        # Ensure predictions are 1D (squeeze all dimensions)
        predictions_array = np.squeeze(predictions_array)

        # If still multi-dimensional, flatten
        if predictions_array.ndim > 1:
            predictions_array = predictions_array.flatten()

        # Denormalize predictions (assuming max_milk=80.0 from config)
        predictions_array = predictions_array * 80.0

        return cls(
            predictions=predictions_array,
            input_data=input_data,
            metadata=metadata,
        )

    def attach_actual_labels(self, item: dict[str, object]) -> None:
        """
        Attach actual labels from a dataset item for plotting comparison.

        Args:
            item: Dataset item with 'labels' or 'features' containing milk data
        """
        if "labels" in item:
            self.input_data = {"milk": item["labels"]}
        elif "features" in item:
            features = item["features"]
            if isinstance(features, dict) and "milk" in features:
                self.input_data = {"milk": features["milk"]}

    def get_actual_labels(self) -> npt.NDArray[np.float32] | None:
        """
        Retrieve actual milk values (ground truth) from input data.

        For the autoencoder, the labels are the actual milk sequence that
        the model tries to reconstruct.

        Returns:
            Actual milk values (denormalized to kg) if available, otherwise None
        """
        if self.input_data is None or "milk" not in self.input_data:
            return None

        milk = self.input_data["milk"]
        if isinstance(milk, np.ndarray):
            milk_array: npt.NDArray[np.float32] = milk.astype(np.float32)
            # Denormalize if normalized (max_milk=80.0 from config)
            if np.max(milk_array) <= 1.0:
                milk_array = milk_array * 80.0
            return milk_array

        return None

    def to_serializable(self) -> dict[str, object]:
        """
        Convert to serializable dict for MLflow signatures.

        Returns:
            Dict with predictions and metadata
        """
        return {
            "predictions": self.predictions.tolist(),
            "metadata": self.metadata,
            "shape": list(self.predictions.shape),
        }

    def get_prediction_stats(self) -> dict[str, float]:
        """
        Calculate prediction statistics.

        Returns:
            Dict with mean, max, min, std of predictions
        """
        return {
            "mean_milk": float(np.mean(self.predictions)),
            "max_milk": float(np.max(self.predictions)),
            "min_milk": float(np.min(self.predictions)),
            "std_milk": float(np.std(self.predictions)),
            "total_milk": float(np.sum(self.predictions)),
        }

    def plot_prediction(self, ax: Axes | None = None, show_input: bool = True) -> Axes:
        """
        Plot the prediction curve with optional actual data.

        Args:
            ax: Matplotlib axis (if None, creates new figure)
            show_input: Whether to plot actual milk data alongside prediction

        Returns:
            Matplotlib axis
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        # Flatten predictions to ensure 1D
        predictions_flat = self.predictions.flatten()
        days = np.arange(len(predictions_flat))
        ax.plot(days, predictions_flat, label="Predicted", linewidth=2)

        # Plot actual data if available
        if show_input:
            actual_milk = self.get_actual_labels()
            if actual_milk is not None:
                actual_flat = actual_milk.flatten()
                ax.plot(
                    days[: len(actual_flat)], actual_flat, label="Actual", alpha=0.7, linewidth=1.5
                )

        ax.set_xlabel("Days in Lactation")
        ax.set_ylabel("Milk Production (kg)")
        ax.set_title("Lactation Curve Prediction")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_error_analysis(self, ax: Axes | None = None) -> Axes | None:
        """
        Plot prediction error (residuals) over time.

        Args:
            ax: Matplotlib axis (if None, creates new figure)

        Returns:
            Matplotlib axis or None if no actual labels available
        """
        import matplotlib.pyplot as plt

        # Get actual milk values using the helper method
        actual_milk = self.get_actual_labels()
        if actual_milk is None:
            print("No actual labels available for error analysis")
            return None

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        # Flatten arrays to ensure 1D
        predictions_flat = self.predictions.flatten()
        actual_flat = actual_milk.flatten()

        # Calculate residuals
        min_len = min(len(predictions_flat), len(actual_flat))
        residuals = predictions_flat[:min_len] - actual_flat[:min_len]
        days = np.arange(min_len)

        # Plot residuals
        ax.plot(days, residuals, label="Prediction Error", linewidth=1.5, color="red")
        ax.fill_between(days, 0, residuals, alpha=0.3, color="red")
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Days in Lactation")
        ax.set_ylabel("Prediction Error (kg)")
        ax.set_title("Prediction Error Analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax
