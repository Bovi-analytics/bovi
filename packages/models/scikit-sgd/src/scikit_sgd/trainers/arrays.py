"""Conversion helpers shared by the trainer and evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from bovi_core.ml import AbstractDataLoader

from scikit_sgd.models import ScikitSGDModel


def batch_to_arrays(
    batch: Mapping[str, Any],
    feature_names: tuple[str, ...],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    features = batch.get("features")
    if not isinstance(features, Mapping):
        raise TypeError("Scikit SGD batches require a 'features' mapping")

    try:
        columns = [
            np.asarray(features[name], dtype=np.float64).reshape(-1) for name in feature_names
        ]
    except KeyError as exc:
        raise ValueError(f"Batch is missing configured feature: {exc.args[0]}") from exc

    labels = batch.get("labels")
    if labels is None:
        raise ValueError("Scikit SGD batches require labels")

    x = np.column_stack(columns)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Feature and label batch sizes do not match")
    return x, y


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
