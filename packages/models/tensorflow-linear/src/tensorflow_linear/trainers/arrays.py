"""Validate batches and calculate scalar regression metrics."""

from bovi_core.ml.dataloaders.model_inputs.tensorflow_regression import (
    prepare_tensorflow_regression_inputs as batch_to_arrays,
)
from bovi_core.ml.trainers.monitoring import RegressionMetrics


def measure(model, loader):
    metrics = RegressionMetrics()
    for batch in loader:
        x, y = batch_to_arrays(batch, model.config.feature_names)
        metrics.update(y.numpy(), model(x))
    return metrics.result()
