"""Independent evaluation, including models restored without a training result."""

from bovi_core.ml import (
    AbstractDataLoader,
    EvaluationContext,
    EvaluationResult,
    Evaluator,
)
from bovi_core.ml.trainers.lifecycle import run_evaluation

from ..models import TensorFlowLinearModel
from .arrays import measure
from .config import TensorFlowLinearEvaluationConfig


class TensorFlowLinearEvaluator(Evaluator[TensorFlowLinearModel, TensorFlowLinearEvaluationConfig]):
    def evaluate(
        self, dataloader: AbstractDataLoader, context: EvaluationContext
    ) -> EvaluationResult:
        def selected_metrics(batches):
            count, metrics = measure(self.model, batches)
            return count, {name: metrics[name] for name in self.config.metrics}

        return run_evaluation(
            context=context,
            dataloader=dataloader,
            measure=selected_metrics,
            error_code="tensorflow_linear_evaluation_failed",
        )
