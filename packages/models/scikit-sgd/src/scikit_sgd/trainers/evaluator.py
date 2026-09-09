"""Concrete evaluator for scikit SGD regression models."""

from __future__ import annotations

from bovi_core.ml import (
    AbstractDataLoader,
    EvaluationContext,
    EvaluationResult,
    Evaluator,
)
from bovi_core.ml.trainers.lifecycle import run_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scikit_sgd.models import ScikitSGDModel

from .arrays import collect_predictions, measure
from .config import ScikitSGDEvaluationConfig


class ScikitSGDEvaluator(Evaluator[ScikitSGDModel, ScikitSGDEvaluationConfig]):
    """Evaluate a fitted SGD regressor against one named data split."""

    def evaluate(
        self,
        dataloader: AbstractDataLoader,
        context: EvaluationContext,
    ) -> EvaluationResult:
        def selected_metrics(batches):
            if "r2" not in self.config.metrics:
                count, metrics = measure(self.model, batches)
            else:
                expected, predicted = collect_predictions(
                    self.model, batches, self.model.config.feature_names
                )
                metric_functions = {
                    "mse": mean_squared_error,
                    "mae": mean_absolute_error,
                    "r2": r2_score,
                }
                count = len(expected)
                metrics = {
                    name: float(metric_functions[name](expected, predicted))
                    for name in self.config.metrics
                }
            return count, {name: metrics[name] for name in self.config.metrics}

        return run_evaluation(
            context=context,
            dataloader=dataloader,
            measure=selected_metrics,
            error_code="scikit_sgd_evaluation_failed",
        )
