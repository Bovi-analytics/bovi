"""Optional execution helpers; native training and metric calculation stay concrete."""

from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime
from typing import Any, TypeVar
from uuid import uuid4

from .context import EvaluationContext, ExecutionContext, TrainingContext
from .evaluation import EvaluationResult, EvaluationStatus
from .issues import Issue, IssueSeverity
from .monitoring import MetricMonitor
from .results import EpochResult, TrainingResult, TrainingStatus, TrainingStopReason

BatchT = TypeVar("BatchT")


class DeadlineReached(Exception):
    """A cooperative deadline boundary was reached, not a hard timeout."""


def check_deadline(context: ExecutionContext | None) -> None:
    if context is not None and context.deadline is not None:
        if datetime.now(UTC) >= context.deadline:
            raise DeadlineReached


def deadline_batches(
    batches: Iterable[BatchT], context: ExecutionContext | None
) -> Iterator[BatchT]:
    """Check before fetching and processing each batch, including the last one."""
    check_deadline(context)
    iterator = iter(batches)
    while True:
        check_deadline(context)
        try:
            batch = next(iterator)
        except StopIteration:
            return
        check_deadline(context)
        yield batch


def run_epochs(
    *,
    context: TrainingContext | None,
    dataloaders: Mapping[str, Any],
    epochs: int,
    monitor: MetricMonitor,
    monitor_metric: str,
    prepare: Callable[[], Callable[[Any], int]],
    measure: Callable[[Iterable[Any]], tuple[int, dict[str, float]]],
    save: Callable[..., Any],
    error_code: str,
) -> TrainingResult:
    """Run optional epoch bookkeeping around a native step returning exposure count.

    Metrics re-evaluate the updated model, rather than averaging training-step losses.
    Only completed metric epochs are eligible for checkpoint references.
    """
    started = datetime.now(UTC)
    history: list[EpochResult] = []
    best_epoch = None
    last_checkpoint = best_checkpoint = None
    status, stop = TrainingStatus.COMPLETED, TrainingStopReason.MAX_EPOCHS_REACHED
    issues = ()
    num_examples = None
    processed = 0
    try:
        train = dataloaders["train"]
        dataset = getattr(train, "dataset", None)
        if dataset is not None:
            try:
                num_examples = len(dataset)
            except TypeError:
                pass
        check_deadline(context)
        step = prepare()
        for epoch in range(1, epochs + 1):
            check_deadline(context)
            # Explicit epochs keep metric passes from advancing training shuffle state.
            for loader in dataloaders.values():
                set_epoch = getattr(loader, "set_epoch", None)
                if set_epoch is not None:
                    if getattr(loader, "shuffle", False) and getattr(loader, "seed", 0) is None:
                        continue
                    set_epoch(epoch - 1)
            for batch in deadline_batches(train, context):
                processed += step(batch)
            _, train_metrics = measure(deadline_batches(train, context))
            metrics = {"train_" + name: value for name, value in train_metrics.items()}
            monitored = train_metrics[monitor_metric]
            validation = dataloaders.get("validation")
            if validation is not None:
                _, validation_metrics = measure(deadline_batches(validation, context))
                metrics.update({"validation_" + k: v for k, v in validation_metrics.items()})
                monitored = validation_metrics[monitor_metric]
            check_deadline(context)
            history.append(EpochResult(epoch=epoch, metrics=metrics))
            last_checkpoint = save("last", epoch=epoch)
            improved, reason = monitor.observe(monitored)
            if improved:
                best_checkpoint = save("best", epoch=epoch)
                best_epoch = epoch
            if reason is not None:
                stop = reason
                break
    except DeadlineReached:
        status, stop = TrainingStatus.CANCELLED, TrainingStopReason.DEADLINE_REACHED
        issues = (
            Issue(
                severity=IssueSeverity.WARNING,
                code="checkpoint_not_current_model",
                message="last_checkpoint is the prior completed epoch, if available; "
                "in-memory weights may include partial epoch updates.",
            ),
        )
    except Exception as exc:
        status, stop = TrainingStatus.FAILED, TrainingStopReason.ERROR
        issues = (Issue.from_exception(exc, code=error_code),)
    return TrainingResult(
        run_id=context.run_id if context else uuid4(),
        status=status,
        stop_reason=stop,
        started_at=started,
        completed_at=datetime.now(UTC),
        epochs=tuple(history),
        best_epoch=best_epoch,
        last_checkpoint=last_checkpoint,
        best_checkpoint=best_checkpoint,
        issues=issues,
        num_examples=num_examples,
        num_examples_processed=processed,
    )


def run_evaluation(
    *,
    context: EvaluationContext,
    dataloader: Iterable[Any],
    measure: Callable[[Iterable[Any]], tuple[int, dict[str, float]]],
    error_code: str,
) -> EvaluationResult:
    """Share evaluation status/error handling without constraining metric selection."""
    started = datetime.now(UTC)
    try:
        check_deadline(context)
        count, metrics = measure(deadline_batches(dataloader, context))
        check_deadline(context)
        return EvaluationResult(
            evaluation_id=context.evaluation_id,
            status=EvaluationStatus.COMPLETED,
            started_at=started,
            completed_at=datetime.now(UTC),
            num_examples=count,
            metrics=metrics,
        )
    except DeadlineReached:
        status, issues = EvaluationStatus.CANCELLED, ()
    except Exception as exc:
        status = EvaluationStatus.FAILED
        issues = (Issue.from_exception(exc, code=error_code),)
    return EvaluationResult(
        evaluation_id=context.evaluation_id,
        status=status,
        started_at=started,
        completed_at=datetime.now(UTC),
        num_examples=0,
        issues=issues,
    )
