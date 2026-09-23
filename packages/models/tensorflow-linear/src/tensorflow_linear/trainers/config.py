"""Immutable controls for one training or evaluation attempt."""

from typing import ClassVar, Literal

from bovi_core.ml import EvaluationConfig, TrainingConfig
from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt


class TensorFlowLinearTrainingConfig(TrainingConfig):
    model_key: ClassVar[str] = "tensorflow_linear"
    epochs: PositiveInt = 40
    learning_rate: PositiveFloat = 0.1
    early_stopping_patience: PositiveInt | None = None
    min_delta: NonNegativeFloat = 0.0
    target_mse: NonNegativeFloat | None = None


class TensorFlowLinearEvaluationConfig(EvaluationConfig):
    model_key: ClassVar[str] = "tensorflow_linear"
    metrics: tuple[Literal["mse", "mae"], ...] = Field(default=("mse", "mae"), min_length=1)
