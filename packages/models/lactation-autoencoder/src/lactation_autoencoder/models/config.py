"""Typed construction configuration for the lactation autoencoder."""

from typing import ClassVar, Literal

from bovi_core.ml import ModelConfig
from pydantic import Field


class LactationAutoencoderModelConfig(ModelConfig):
    """Architecture and serving settings required to load the model."""

    model_key: ClassVar[str] = "autoencoder"
    framework: Literal["tensorflow"] = "tensorflow"

    input_dim: int = Field(gt=0)
    latent_dim: int = Field(gt=0)
    num_events: int = Field(gt=0)
    num_herd_stats: int = Field(gt=0)
    signature_name: str = Field(default="serving_default", min_length=1)
