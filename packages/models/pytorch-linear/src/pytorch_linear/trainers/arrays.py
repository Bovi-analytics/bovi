"""Validate batches and calculate scalar regression metrics."""

from bovi_core.ml.dataloaders.adapters.tabular import pytorch_regression_batch as batch_to_arrays
from bovi_core.ml.trainers.monitoring import RegressionMetrics


def measure(model, loader):
    metrics = RegressionMetrics()
    for batch in loader:
        x, y = batch_to_arrays(batch, model.config.feature_names)
        metrics.update(y.detach().cpu().numpy(), model(x))
    return metrics.result()
