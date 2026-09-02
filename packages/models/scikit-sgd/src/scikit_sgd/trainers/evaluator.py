"""Concrete evaluator for scikit SGD regression models."""

from __future__ import annotations

from datetime import UTC, datetime

from bovi_core.ml import (
    AbstractDataLoader,
    EvaluationContext,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
    Issue,
    IssueSeverity,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scikit_sgd.models import ScikitSGDModel

from .arrays import collect_predictions
from .config import ScikitSGDEvaluationConfig


class ScikitSGDEvaluator(Evaluator[ScikitSGDModel, ScikitSGDEvaluationConfig]):
    """Evaluate a fitted SGD regressor against one named data split."""

    def evaluate(
        self,
        dataloader: AbstractDataLoader,
        context: EvaluationContext,
    ) -> EvaluationResult:
        started_at = datetime.now(UTC)
        try:
            expected, predicted = collect_predictions(
                self.model,
                dataloader,
                self.model.config.feature_names,
            )
            metric_functions = {
                "mse": mean_squared_error,
                "mae": mean_absolute_error,
                "r2": r2_score,
            }
            metrics = {
                name: float(metric_functions[name](expected, predicted))
                for name in self.config.metrics
            }
            return EvaluationResult(
                evaluation_id=context.evaluation_id,
                status=EvaluationStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                num_examples=len(expected),
                metrics=metrics,
            )
        except Exception as exc:
            issue = Issue(
                severity=IssueSeverity.ERROR,
                code="scikit_sgd_evaluation_failed",
                message=str(exc),
                exception_type=type(exc).__name__,
            )
            return EvaluationResult(
                evaluation_id=context.evaluation_id,
                status=EvaluationStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                num_examples=0,
                issues=(issue,),
            )
