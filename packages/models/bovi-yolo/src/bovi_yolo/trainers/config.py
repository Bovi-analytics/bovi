"""Immutable Ultralytics training and evaluation controls."""

from pathlib import Path
from typing import ClassVar, Literal

from bovi_core.ml import EvaluationConfig, TrainingConfig
from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt


class YOLOTrainingConfig(TrainingConfig):
    """Configuration for one native Ultralytics detection training run."""

    model_key: ClassVar[str] = "yolo"

    dataset_yaml_path: Path
    epochs: PositiveInt = 1
    image_size: PositiveInt = 640
    batch_size: PositiveInt = 16
    device: str = "cpu"
    workers: NonNegativeInt = 0
    patience: NonNegativeInt = 100
    optimizer: str = "auto"
    initial_learning_rate: PositiveFloat = 0.01
    seed: NonNegativeInt = 0
    deterministic: bool = True
    reuse_model_weights: bool = True
    resume: bool = False
    plots: bool = False
    verbose: bool = False


class YOLOEvaluationConfig(EvaluationConfig):
    """Configuration for one native Ultralytics detection evaluation."""

    model_key: ClassVar[str] = "yolo"

    dataset_yaml_path: Path
    split: Literal["val", "test"] = "val"
    image_size: PositiveInt = 640
    batch_size: PositiveInt = 16
    device: str = "cpu"
    workers: NonNegativeInt = 0
    confidence_threshold: NonNegativeFloat = Field(default=0.001, le=1)
    iou_threshold: NonNegativeFloat = Field(default=0.7, le=1)
    metrics: tuple[Literal["precision", "recall", "map50", "map50_95"], ...] = Field(
        default=("precision", "recall", "map50", "map50_95"), min_length=1
    )
    plots: bool = False
    verbose: bool = False
