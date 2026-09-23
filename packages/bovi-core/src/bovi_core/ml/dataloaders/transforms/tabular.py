"""Generic stateless transforms for numeric tabular fields."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from bovi_core.ml.dataloaders.transforms.base_transform import UniversalTransform
from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry


def _apply_to_fields(
    data: Mapping[str, Any],
    operations: Mapping[str, Callable[[Any], Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            result[key] = _apply_to_fields(value, operations)
        elif key in operations:
            result[key] = operations[key](value)
        else:
            result[key] = value
    return result


def _numeric_array(value: Any, field: str) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"Field '{field}' must contain numeric values")
    return array


def _restore_scalar(value: Any, transformed: np.ndarray[Any, Any]) -> Any:
    if np.isscalar(value):
        return transformed.item()
    return transformed


@TransformRegistry.register("numeric_clip")
class NumericClipTransform(UniversalTransform):
    """Clip configured numeric fields to inclusive lower and upper bounds."""

    def __init__(self, ranges: Mapping[str, Sequence[float]]) -> None:
        self.ranges: dict[str, tuple[float, float]] = {}
        for field, bounds in ranges.items():
            if len(bounds) != 2:
                raise ValueError(f"Range for '{field}' must contain exactly two bounds")
            lower, upper = float(bounds[0]), float(bounds[1])
            if lower > upper:
                raise ValueError(f"Lower bound for '{field}' cannot exceed upper bound")
            self.ranges[field] = (lower, upper)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        operations = {
            field: self._clipper(field, lower, upper)
            for field, (lower, upper) in self.ranges.items()
        }
        return _apply_to_fields(data, operations)

    @staticmethod
    def _clipper(field: str, lower: float, upper: float) -> Callable[[Any], Any]:
        def clip(value: Any) -> Any:
            transformed = np.clip(_numeric_array(value, field), lower, upper)
            return _restore_scalar(value, transformed)

        return clip

    def get_params(self) -> dict[str, object]:
        return {"ranges": self.ranges.copy()}


@TransformRegistry.register("numeric_scale")
class NumericScaleTransform(UniversalTransform):
    """Divide configured numeric fields by fixed, non-zero scale factors."""

    def __init__(self, factors: Mapping[str, float]) -> None:
        self.factors = {field: float(factor) for field, factor in factors.items()}
        zero_fields = [field for field, factor in self.factors.items() if factor == 0.0]
        if zero_fields:
            fields = ", ".join(sorted(zero_fields))
            raise ValueError(f"Scale factors must be non-zero for fields: {fields}")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        operations = {field: self._scaler(field, factor) for field, factor in self.factors.items()}
        return _apply_to_fields(data, operations)

    @staticmethod
    def _scaler(field: str, factor: float) -> Callable[[Any], Any]:
        def scale(value: Any) -> Any:
            transformed = _numeric_array(value, field) / factor
            return _restore_scalar(value, transformed)

        return scale

    def get_params(self) -> dict[str, object]:
        return {"factors": self.factors.copy()}
