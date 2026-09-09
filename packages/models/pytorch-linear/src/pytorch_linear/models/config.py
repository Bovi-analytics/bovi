"""Construction settings for the CPU linear regressor."""

from typing import ClassVar, Literal

from bovi_core.ml import ModelConfig
from pydantic import Field


class PyTorchLinearModelConfig(ModelConfig):
    model_key: ClassVar[str] = "pytorch_linear"
    framework: Literal["pytorch"] = "pytorch"
    feature_names: tuple[str, ...] = Field(default=("x",), min_length=1)
