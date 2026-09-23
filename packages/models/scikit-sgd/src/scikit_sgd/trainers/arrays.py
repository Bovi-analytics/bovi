"""Conversion helpers shared by the trainer and evaluator."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from bovi_core.ml import AbstractDataLoader
from bovi_core.ml.dataloaders.model_inputs.numpy_regression import (
    prepare_numpy_regression_inputs as batch_to_arrays,
)
from bovi_core.ml.trainers.monitoring import RegressionMetrics

from scikit_sgd.models import ScikitSGDModel


def measure(model, loader):
    metrics = RegressionMetrics()
    for batch in loader:
        x, y = batch_to_arrays(batch, model.config.feature_names)
        metrics.update(y, model(x))
    return metrics.result()


def collect_predictions(
    model: ScikitSGDModel,
    dataloader: AbstractDataLoader,
    feature_names: tuple[str, ...],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    labels: list[npt.NDArray[np.float64]] = []
    predictions: list[npt.NDArray[np.float64]] = []
    for batch in dataloader:
        x, y = batch_to_arrays(batch, feature_names)
        labels.append(y)
        predictions.append(np.asarray(model(x), dtype=np.float64).reshape(-1))

    if not labels:
        raise ValueError("Cannot evaluate an empty dataloader")
    return np.concatenate(labels), np.concatenate(predictions)
