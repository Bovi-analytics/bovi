"""Bovi evaluator adapter around native Ultralytics validation."""

from datetime import UTC, datetime

from bovi_core.ml import (
    AbstractDataLoader,
    EvaluationContext,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
    Issue,
)
from bovi_core.ml.trainers.lifecycle import DeadlineReached, check_deadline

from ..models import YOLOModel
from .config import YOLOEvaluationConfig
from .native import count_split_images, scalar_metrics


class YOLOEvaluator(Evaluator[YOLOModel, YOLOEvaluationConfig]):
    """Evaluate through Ultralytics because Bovi's inference loader has no targets."""

    def evaluate(
        self,
        dataloader: AbstractDataLoader,
        context: EvaluationContext,
    ) -> EvaluationResult:
        # The Core signature remains uniform, but native detection evaluation must
        # consume labels from dataset_yaml_path instead of this inference loader.
        del dataloader
        started_at = datetime.now(UTC)
        try:
            check_deadline(context)
            native_metrics = self.model.native_model.val(
                data=str(self.config.dataset_yaml_path),
                split=self.config.split,
                imgsz=self.config.image_size,
                batch=self.config.batch_size,
                device=self.config.device,
                workers=self.config.workers,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                plots=self.config.plots,
                verbose=self.config.verbose,
                project=str(context.output_dir / "ultralytics"),
                name=str(context.evaluation_id),
                exist_ok=True,
            )
            check_deadline(context)
            available = scalar_metrics(native_metrics)
            metrics = {name: available[name] for name in self.config.metrics if name in available}
            if not metrics:
                raise ValueError("Ultralytics returned none of the configured evaluation metrics")
            num_examples = count_split_images(
                self.config.dataset_yaml_path,
                self.config.split,
            )
            return EvaluationResult(
                evaluation_id=context.evaluation_id,
                status=EvaluationStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                num_examples=num_examples,
                metrics=metrics,
            )
        except DeadlineReached:
            status = EvaluationStatus.CANCELLED
            issues: tuple[Issue, ...] = ()
        except Exception as exception:
            status = EvaluationStatus.FAILED
            issues = (Issue.from_exception(exception, code="yolo_evaluation_failed"),)
        return EvaluationResult(
            evaluation_id=context.evaluation_id,
            status=status,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            num_examples=0,
            issues=issues,
        )
