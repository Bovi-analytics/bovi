"""Typed construction configuration for Ultralytics YOLO models."""

from typing import ClassVar, Literal

from bovi_core.ml import ModelConfig


class YOLOModelConfig(ModelConfig):
    """Configuration used by :class:`YOLOModelProvider`."""

    model_key: ClassVar[str] = "yolo"
    framework: Literal["pytorch"] = "pytorch"
    model_source: str = "yolo12n.yaml"
    task: Literal["detect", "segment", "classify", "pose", "obb"] = "detect"
