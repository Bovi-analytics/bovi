"""Immutable controls for lactation training and evaluation."""

from typing import ClassVar, Literal

from bovi_core.ml import EvaluationConfig, TrainingConfig
from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt


class LactationAutoencoderTrainingConfig(TrainingConfig):
    model_key: ClassVar[str] = "autoencoder"

    epochs: PositiveInt = 50
    learning_rate: PositiveFloat = 0.001
    optimizer: Literal["adam"] = "adam"
    loss: Literal["mse"] = "mse"
    early_stopping_patience: PositiveInt | None = None
    min_delta: NonNegativeFloat = 0.0
    target_mse: NonNegativeFloat | None = None


class LactationAutoencoderEvaluationConfig(EvaluationConfig):
    model_key: ClassVar[str] = "autoencoder"

    metrics: tuple[Literal["mse", "mae", "rmse"], ...] = Field(
        default=("mse", "mae", "rmse"), min_length=1
    )
