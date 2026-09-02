"""Typed runtime configuration for scikit SGD training and evaluation."""

from typing import ClassVar, Literal

from bovi_core.ml import EvaluationConfig, TrainingConfig
from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt


class ScikitSGDTrainingConfig(TrainingConfig):
    """Runtime controls for one local SGD training attempt."""

    model_key: ClassVar[str] = "scikit_sgd"
    epochs: PositiveInt
    learning_rate: PositiveFloat
    learning_rate_schedule: Literal["constant", "optimal", "invscaling", "adaptive"] = "constant"
    loss: Literal[
        "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"
    ] = "squared_error"
    penalty: Literal["l2", "l1", "elasticnet"] | None = "l2"
    alpha: NonNegativeFloat = 0.0001
    average: bool = False
    early_stopping_patience: PositiveInt | None = None
    min_delta: NonNegativeFloat = 0.0
    target_mse: NonNegativeFloat | None = None


class ScikitSGDEvaluationConfig(EvaluationConfig):
    """Scalar regression metrics to calculate for an evaluation."""

    model_key: ClassVar[str] = "scikit_sgd"
    metrics: tuple[Literal["mse", "mae", "r2"], ...] = Field(min_length=1)
