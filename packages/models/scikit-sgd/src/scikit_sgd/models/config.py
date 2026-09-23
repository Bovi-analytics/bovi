"""Typed configuration for the scikit SGD model."""

from typing import ClassVar, Literal

from bovi_core.ml import ModelConfig
from pydantic import Field


class ScikitSGDModelConfig(ModelConfig):
    """Static input and estimator settings shared by training and inference."""

    model_key: ClassVar[str] = "scikit_sgd"
    framework: Literal["sklearn"] = "sklearn"
    feature_names: tuple[str, ...] = Field(min_length=1)
    fit_intercept: bool = True
    random_state: int = 42
