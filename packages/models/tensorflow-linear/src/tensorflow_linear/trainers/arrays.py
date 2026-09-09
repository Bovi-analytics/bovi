"""Validate batches and calculate scalar regression metrics."""

import numpy as np


def batch_to_arrays(batch, num_features: int):
    x = np.asarray(batch["features"], dtype=np.float32)
    y = np.asarray(batch["labels"], dtype=np.float32).reshape(-1)
    if x.ndim != 2 or x.shape != (len(y), num_features):
        raise ValueError("Feature and label shapes do not match the model")
    if not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Batches must contain finite, nonempty regression data")
    return x, y


def measure(model, loader):
    expected, predicted = [], []
    for batch in loader:
        x, y = batch_to_arrays(batch, len(model.config.feature_names))
        expected.append(y)
        predicted.append(model(x))
    if not expected:
        raise ValueError("Cannot evaluate an empty dataloader")
    y = np.concatenate(expected)
    errors = np.concatenate(predicted) - y
    return len(y), {"mse": float(np.mean(errors**2)), "mae": float(np.mean(np.abs(errors)))}
