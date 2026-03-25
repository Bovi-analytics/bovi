"""YOLO models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .yolo_model import YOLOModel

if TYPE_CHECKING:
    from .yolo_unity_catalog import YOLOModelWrapper


def __getattr__(name: str) -> type:
    """Lazy import for YOLOModelWrapper to avoid requiring mlflow at import time."""
    if name == "YOLOModelWrapper":
        from .yolo_unity_catalog import YOLOModelWrapper

        return YOLOModelWrapper
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "YOLOModel",
    "YOLOModelWrapper",
]
