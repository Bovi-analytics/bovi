"""Independent evaluation, including models restored without a training result."""

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

from ..models import PyTorchLinearModel
from .arrays import measure
from .config import PyTorchLinearEvaluationConfig


class PyTorchLinearEvaluator(Evaluator[PyTorchLinearModel, PyTorchLinearEvaluationConfig]):
    def evaluate(
        self, dataloader: AbstractDataLoader, context: EvaluationContext
    ) -> EvaluationResult:
        started = datetime.now(UTC)
        if context.deadline is not None and started >= context.deadline:
            return EvaluationResult(
                evaluation_id=context.evaluation_id,
                status=EvaluationStatus.CANCELLED,
                started_at=started,
                completed_at=datetime.now(UTC),
                num_examples=0,
            )
        try:
            count, metrics = measure(self.model, dataloader)
            return EvaluationResult(
                evaluation_id=context.evaluation_id,
                status=EvaluationStatus.COMPLETED,
                started_at=started,
                completed_at=datetime.now(UTC),
                num_examples=count,
                metrics={name: metrics[name] for name in self.config.metrics},
            )
        except Exception as exc:
            return EvaluationResult(
                evaluation_id=context.evaluation_id,
                status=EvaluationStatus.FAILED,
                started_at=started,
                completed_at=datetime.now(UTC),
                num_examples=0,
                issues=(
                    Issue(
                        severity=IssueSeverity.ERROR,
                        code="pytorch_linear_evaluation_failed",
                        message=str(exc),
                        exception_type=type(exc).__name__,
                    ),
                ),
            )
