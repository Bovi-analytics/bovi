"""
Time-series transforms: Imputation, Normalization, Padding, etc.

Generic transforms for any time-series data (not specific to domain).
These are atomic, config-driven transforms that can be composed into pipelines.

All transforms are STATELESS - they process data per-sample without requiring fit().
"""

from __future__ import annotations

import logging

import numpy as np

from bovi_core.ml.dataloaders.base.universal_transform import UniversalTransform
from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry

logger = logging.getLogger(__name__)


@TransformRegistry.register("imputation")
class ImputationTransform(UniversalTransform):
    """
    Generic imputation for missing values in sequences.

    Supports multiple methods:
    - forward_fill: Copy last valid value forward
    - backward_fill: Copy next valid value backward
    - linear: Linear interpolation
    - zero: Replace with 0
    - mean: Replace with field mean (computed per-sample, stateless)

    Works with any numeric sequence, not just milk data.
    """

    # Type annotations for instance attributes
    method: str
    fields: list[str] | None

    def __init__(
        self,
        method: str = "forward_fill",
        fields: list[str] | None = None,
    ) -> None:
        """
        Args:
            method: Imputation method (forward_fill, backward_fill, linear, zero, mean).
            fields: List of field names to impute (None = auto-detect numeric fields).
        """
        if method not in ["forward_fill", "backward_fill", "linear", "zero", "mean"]:
            raise ValueError(f"Unknown imputation method: {method}")
        self.method = method
        self.fields = fields

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """Apply imputation to data dict."""
        return self._apply_imputation(data)

    def _apply_imputation(self, data: dict[str, object]) -> dict[str, object]:
        """
        Apply imputation to specified fields.

        Args:
            data: Dict with fields potentially containing missing values.

        Returns:
            Dict with imputed values.
        """
        data = data.copy()

        # Recursively process nested dicts and arrays
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively process nested dict
                data[key] = self._apply_imputation(value)
            elif isinstance(value, (list, np.ndarray)):
                # Check if this field should be processed
                if self.fields is not None and key not in self.fields:
                    continue

                # Check if this is a numeric array (may contain None values)
                try:
                    arr = np.asarray(value)
                    is_numeric = False

                    if arr.dtype.kind in ["i", "f", "u"]:
                        # Pure numeric array
                        is_numeric = True
                    elif arr.dtype == object and len(arr) > 0:
                        # Object array - check if first non-None element is numeric
                        # This handles arrays like [25.3, None, 27.5, ...]
                        for elem in arr:
                            if elem is not None:
                                is_numeric = isinstance(elem, (int, float, np.integer, np.floating))
                                break

                    if is_numeric:
                        data[key] = self._impute_array(arr)
                except (ValueError, TypeError):
                    # Not a numeric array, skip
                    pass

        return data

    def _impute_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Impute single array.

        Args:
            arr: Array possibly containing None, np.nan, etc.

        Returns:
            Imputed array.
        """
        # Convert to float for easier handling
        if arr.dtype == object:
            arr_float = np.empty(len(arr), dtype=float)
            for i, v in enumerate(arr):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    arr_float[i] = np.nan
                else:
                    arr_float[i] = float(v)
        else:
            arr_float = arr.astype(float)

        # Replace inf with nan
        arr_float = np.where(np.isinf(arr_float), np.nan, arr_float)

        if np.isnan(arr_float).sum() == 0:
            # No missing values
            return arr_float

        # Apply method
        if self.method == "forward_fill":
            return self._forward_fill(arr_float)
        elif self.method == "backward_fill":
            return self._backward_fill(arr_float)
        elif self.method == "linear":
            return self._linear_interpolate(arr_float)
        elif self.method == "zero":
            return np.nan_to_num(arr_float, nan=0.0)
        elif self.method == "mean":
            mean_val = float(np.nanmean(arr_float))
            return np.nan_to_num(arr_float, nan=mean_val)
        else:
            return arr_float

    @staticmethod
    def _forward_fill(arr: np.ndarray) -> np.ndarray:
        """Forward fill missing values."""
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        return arr[idx]

    @staticmethod
    def _backward_fill(arr: np.ndarray) -> np.ndarray:
        """Backward fill missing values."""
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(len(arr)), len(arr) - 1)
        idx = np.minimum.accumulate(idx[::-1])[::-1]
        return arr[idx]

    @staticmethod
    def _linear_interpolate(arr: np.ndarray) -> np.ndarray:
        """Linear interpolation of missing values."""
        mask = np.isnan(arr)
        idx = np.arange(len(arr))
        arr[mask] = np.interp(idx[mask], idx[~mask], arr[~mask])
        return arr

    def get_params(self) -> dict[str, object]:
        """
        Get transform parameters.

        Returns:
            Dict of parameters.
        """
        return {
            "name": "imputation",
            "method": self.method,
            "fields": self.fields,
        }


@TransformRegistry.register("sequence_normalization")
class SequenceNormalizationTransform(UniversalTransform):
    """
    Generic normalization for sequences.

    Supports:
    - zscore: (x - mean) / std
    - minmax: (x - min) / (max - min)
    - maxabs: x / max(abs(x))
    - scale: x / scalar

    All methods are stateless - computed per-sample.
    """

    # Type annotations for instance attributes
    method: str
    field: str | None
    scale: float | None
    per_sequence: bool

    def __init__(
        self,
        method: str = "zscore",
        field: str | None = None,
        scale: float | None = None,
        per_sequence: bool = True,
    ) -> None:
        """
        Args:
            method: Normalization method.
            field: Field name to normalize (None = all numeric fields).
            scale: Scalar for 'scale' method.
            per_sequence: If True, normalize per sequence. If False, use global stats.
        """
        if method not in ["zscore", "minmax", "maxabs", "scale"]:
            raise ValueError(f"Unknown normalization method: {method}")
        self.method = method
        self.field = field
        self.scale = scale
        self.per_sequence = per_sequence

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """Apply normalization to data dict."""
        return self._apply_normalization(data)

    def _apply_normalization(self, data: dict[str, object]) -> dict[str, object]:
        """Apply normalization to specified fields."""
        data = data.copy()

        # Recursively process nested dicts and arrays
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively process nested dict
                data[key] = self._apply_normalization(value)
            elif isinstance(value, (list, np.ndarray)):
                # Check if this is a numeric array
                try:
                    arr = np.asarray(value, dtype=float)
                    if arr.dtype.kind in ["i", "f", "u"] and len(arr) > 0:
                        # This is a numeric array - apply normalization
                        if self.field is None or key == self.field:
                            data[key] = self._normalize_array(arr)
                except (ValueError, TypeError):
                    # Not a numeric array, skip
                    pass

        return data

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Normalize single array."""
        arr = np.asarray(arr, dtype=float)

        if self.method == "zscore":
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0:
                return arr - mean
            return (arr - mean) / std

        elif self.method == "minmax":
            min_val = np.min(arr)
            max_val = np.max(arr)
            if max_val - min_val == 0:
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)

        elif self.method == "maxabs":
            max_abs = np.max(np.abs(arr))
            if max_abs == 0:
                return arr
            return arr / max_abs

        elif self.method == "scale":
            if self.scale is None:
                raise ValueError("'scale' method requires 'scale' parameter")
            return arr / self.scale

        return arr

    def get_params(self) -> dict[str, object]:
        """
        Get transform parameters.

        Returns:
            Dict of parameters.
        """
        return {
            "name": "sequence_normalization",
            "method": self.method,
            "field": self.field,
            "scale": self.scale,
            "per_sequence": self.per_sequence,
        }


@TransformRegistry.register("sequence_padding")
class SequencePaddingTransform(UniversalTransform):
    """
    Pad or truncate sequences to fixed length.

    Generic for any sequence data.
    """

    # Type annotations for instance attributes
    max_length: int
    field: str
    pad_value: float
    mode: str

    def __init__(
        self,
        max_length: int,
        field: str,
        pad_value: float = 0.0,
        mode: str = "post",
    ) -> None:
        """
        Args:
            max_length: Target sequence length.
            field: Field to pad/truncate.
            pad_value: Value to use for padding.
            mode: 'post' (pad at end) or 'pre' (pad at start).
        """
        if mode not in ["post", "pre"]:
            raise ValueError(f"Unknown padding mode: {mode}")
        self.max_length = max_length
        self.field = field
        self.pad_value = pad_value
        self.mode = mode

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """Apply padding to data dict."""
        return self._apply_padding(data)

    def _apply_padding(self, data: dict[str, object]) -> dict[str, object]:
        """Apply padding to specified field."""
        data = data.copy()

        # Recursively process nested dicts
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively process nested dict
                data[key] = self._apply_padding(value)
            elif key == self.field and isinstance(value, (list, np.ndarray)):
                # This is the target field - apply padding
                data[key] = self._pad_array(np.asarray(value))

        return data

    def _pad_array(self, arr: np.ndarray) -> np.ndarray:
        """Pad or truncate array to max_length."""
        arr = np.asarray(arr)

        if len(arr) >= self.max_length:
            # Truncate
            return arr[: self.max_length]
        else:
            # Pad
            pad_width = self.max_length - len(arr)

            if self.mode == "post":
                pad_config = (0, pad_width)
            else:  # pre
                pad_config = (pad_width, 0)

            return np.pad(arr, pad_config, constant_values=self.pad_value)

    def get_params(self) -> dict[str, object]:
        """
        Get transform parameters.

        Returns:
            Dict of parameters.
        """
        return {
            "name": "sequence_padding",
            "max_length": self.max_length,
            "field": self.field,
            "pad_value": self.pad_value,
            "mode": self.mode,
        }


@TransformRegistry.register("windowing")
class WindowingTransform(UniversalTransform):
    """
    Create sliding windows from sequences.

    Useful for creating multiple training samples from long sequences.
    """

    # Type annotations for instance attributes
    window_size: int
    stride: int
    field: str | None

    def __init__(
        self,
        window_size: int,
        stride: int = 1,
        field: str | None = None,
    ) -> None:
        """
        Args:
            window_size: Size of each window.
            stride: Step size between windows.
            field: Field to window (None = all array fields).
        """
        self.window_size = window_size
        self.stride = stride
        self.field = field

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """Apply windowing to data dict."""
        return self._apply_windowing(data)

    def _apply_windowing(self, data: dict[str, object]) -> dict[str, object]:
        """
        Apply windowing.

        Note: This is typically used in data preparation, not in the transform pipeline.
        Returns the first window for now.
        """
        # For single-item processing, just return first window
        data = data.copy()

        # Recursively process nested dicts and arrays
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively process nested dict
                data[key] = self._apply_windowing(value)
            elif isinstance(value, (list, np.ndarray)):
                arr = np.asarray(value)
                if len(arr) >= self.window_size:
                    # Apply windowing if:
                    # 1. self.field is None (window all fields)
                    # 2. OR key matches self.field
                    if self.field is None or key == self.field:
                        data[key] = arr[: self.window_size]

        return data

    def get_params(self) -> dict[str, object]:
        """
        Get transform parameters.

        Returns:
            Dict of parameters.
        """
        return {
            "name": "windowing",
            "window_size": self.window_size,
            "stride": self.stride,
            "field": self.field,
        }
