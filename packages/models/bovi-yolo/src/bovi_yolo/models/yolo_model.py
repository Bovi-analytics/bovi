"""Framework-neutral runtime wrapper for an Ultralytics YOLO model."""

from __future__ import annotations

from typing import Any

from bovi_core.ml import Model
from typing_extensions import override
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]

from .yolo_config import YOLOModelConfig


class YOLOModel(Model[YOLO, YOLOModelConfig]):
    """Runtime model containing an already-instantiated Ultralytics model."""

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate inference to the native Ultralytics model."""
        return self.native_model(*args, **kwargs)
