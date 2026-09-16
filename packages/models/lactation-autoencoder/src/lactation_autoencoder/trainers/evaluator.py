"""Independent evaluation for trainable lactation models."""

from bovi_core.ml import AbstractDataLoader, EvaluationContext, EvaluationResult, Evaluator
from bovi_core.ml.trainers.lifecycle import run_evaluation

from lactation_autoencoder.models import LactationAutoencoderModel

from .arrays import measure
from .config import LactationAutoencoderEvaluationConfig


class LactationAutoencoderEvaluator(
    Evaluator[LactationAutoencoderModel, LactationAutoencoderEvaluationConfig]
):
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
            error_code="lactation_autoencoder_evaluation_failed",
        )
