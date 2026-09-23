"""Construction settings for the CPU linear regressor."""

from typing import ClassVar, Literal

from bovi_core.ml import ModelConfig
from pydantic import Field


class TensorFlowLinearModelConfig(ModelConfig):
    model_key: ClassVar[str] = "tensorflow_linear"
    framework: Literal["tensorflow"] = "tensorflow"
    feature_names: tuple[str, ...] = Field(default=("x",), min_length=1)
