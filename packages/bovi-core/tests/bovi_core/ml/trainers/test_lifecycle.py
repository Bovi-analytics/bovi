"""Framework-independent lifecycle, deadline and monitoring regressions."""

from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from bovi_core.ml.trainers import lifecycle
from bovi_core.ml.trainers.context import EvaluationContext, TrainingContext
from bovi_core.ml.trainers.monitoring import MetricMonitor, RegressionMetrics
from bovi_core.ml.trainers.results import CheckpointReference, TrainingResult


def test_raw_best_is_independent_of_patience_significance():
    monitor = MetricMonitor(min_delta=1.0, patience=2)
    assert monitor.observe(10.0) == (True, None)
    assert monitor.observe(9.6) == (True, None)
    assert monitor.observe(9.2) == (True, "early_stopping")
    assert monitor.best == 9.2
    assert monitor.significant_best == 10.0


def test_cumulative_improvement_resets_patience_and_target_takes_priority():
    monitor = MetricMonitor(min_delta=1.0, patience=2, target=8.9)
    monitor.observe(10.0)
    monitor.observe(9.6)
    assert monitor.observe(8.9) == (True, "target_metric_reached")
    assert monitor.stale == 0


def test_regression_metrics_weight_short_final_batch():
    metrics = RegressionMetrics()
    metrics.update([0, 0, 0], [1, 1, 1])
    metrics.update([0], [3])
    assert metrics.result() == (4, {"mse": 3.0, "mae": 1.5})


def test_max_mode_tracks_raw_improvements_and_target():
    monitor = MetricMonitor(mode="max", min_delta=0.1, patience=2, target=0.85)
    assert monitor.observe(0.8) == (True, None)
    assert monitor.observe(0.82) == (True, None)
    assert monitor.observe(0.85) == (True, "target_metric_reached")
    assert monitor.best == 0.85
    assert monitor.significant_best == 0.8


def test_empty_regression_metrics_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        RegressionMetrics().result()


@pytest.mark.parametrize("expected,predicted", [([], []), ([1], [1, 2]), ([1], [float("nan")])])
def test_regression_metrics_reject_invalid_batches(expected, predicted):
    with pytest.raises(ValueError):
        RegressionMetrics().update(expected, predicted)


class Loader:
    dataset = range(3)
    shuffle: bool = False
    seed: int | None = 42

    def __init__(self):
        self.epoch = None
        self.passes = []

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        self.passes.append(self.epoch)
        return iter(([1, 2], [3]))


def execute(
    loader: Loader,
    *,
    step: Callable[[list[int]], int] = len,
    measure: Callable[[Iterable[list[int]]], tuple[int, dict[str, float]]] | None = None,
    save: Callable[..., CheckpointReference | None] | None = None,
    context: TrainingContext | None = None,
    dataloaders: Mapping[str, Loader] | None = None,
    monitor: MetricMonitor | None = None,
    monitor_metric: str = "mse",
    prepare: Callable[[], Callable[[list[int]], int]] | None = None,
) -> TrainingResult:
    return lifecycle.run_epochs(
        context=context,
        dataloaders={"train": loader} if dataloaders is None else dataloaders,
        epochs=3,
        monitor=monitor if monitor is not None else MetricMonitor(min_delta=1.0, patience=2),
        monitor_metric=monitor_metric,
        prepare=prepare if prepare is not None else lambda: step,
        measure=measure or (lambda batches: (sum(map(len, batches)), {"mse": 1.0})),
        save=save or (lambda *args, **kwargs: None),
        error_code="training_failed",
    )


def test_shared_epochs_count_exposures_and_pin_metric_iteration():
    loader = Loader()
    scores = iter([10.0, 9.6, 9.2])
    saves = []

    def measure(batches):
        return sum(map(len, batches)), {"mse": next(scores)}

    result = execute(loader, measure=measure, save=lambda name, epoch: saves.append((name, epoch)))
    assert result.status == "completed"
    assert result.stop_reason == "early_stopping"
    assert result.best_epoch == 3
    assert result.num_examples == 3
    assert result.num_examples_processed == 9
    assert loader.passes == [0, 0, 1, 1, 2, 2]
    assert saves == [(name, epoch) for epoch in (1, 2, 3) for name in ("last", "best")]


def test_save_failure_keeps_completed_metrics_and_successful_exposures():
    def fail(*args, **kwargs):
        raise RuntimeError()

    result = execute(Loader(), save=fail)
    assert result.status == "failed"
    assert len(result.epochs) == 1
    assert result.last_checkpoint is None
    assert result.num_examples_processed == 3
    assert result.issues[0].message == "RuntimeError"


def test_validation_monitor_is_generic_and_does_not_count_eval_exposures():
    scores = iter([0.99, 0.6, 0.99, 0.8])

    def measure(batches):
        return sum(map(len, batches)), {"accuracy": next(scores)}

    result = execute(
        Loader(),
        dataloaders={"train": Loader(), "validation": Loader()},
        measure=measure,
        monitor=MetricMonitor(mode="max", target=0.8),
        monitor_metric="accuracy",
    )
    assert result.status == "completed"
    assert result.stop_reason == "target_metric_reached"
    assert result.best_epoch == 2
    assert result.num_examples_processed == 6


def test_unseeded_shuffle_does_not_require_pinned_epochs():
    loader = Loader()
    loader.shuffle = True
    loader.seed = None
    result = execute(loader)
    assert result.status == "completed"
    assert all(epoch is None for epoch in loader.passes)


def test_missing_training_split_captures_error():
    result = execute(Loader(), dataloaders={})
    assert result.status == "failed"
    assert result.num_examples is None
    assert result.num_examples_processed == 0


def test_preexpired_deadline_does_not_prepare_native_optimizer(tmp_path):
    def prepare():
        pytest.fail("Expired training must not initialize optimizer")

    context = TrainingContext(
        run_id=uuid4(), output_dir=tmp_path, deadline=datetime.now(UTC) - timedelta(seconds=1)
    )
    result = execute(Loader(), context=context, prepare=prepare)
    assert result.status == "cancelled"
    assert result.num_examples == 3
    assert result.num_examples_processed == 0


def test_partial_epoch_deadline_does_not_fetch_or_process_next_batch(tmp_path, monkeypatch):
    now = datetime.now(UTC)
    expired = False
    context = TrainingContext(run_id=uuid4(), output_dir=tmp_path, deadline=now + timedelta(days=1))

    def check(context):
        if expired:
            raise lifecycle.DeadlineReached

    def step(batch):
        nonlocal expired
        expired = True
        return len(batch)

    monkeypatch.setattr(lifecycle, "check_deadline", check)
    result = execute(Loader(), context=context, step=step)
    assert result.status == "cancelled"
    assert result.num_examples_processed == 2
    assert not result.epochs
    assert result.issues[0].code == "checkpoint_not_current_model"


@pytest.mark.parametrize("phase", ["before", "during", "after", "error"])
def test_evaluation_deadline_and_empty_exception_handling(tmp_path, monkeypatch, phase):
    expired = phase == "before"
    visited = []

    def check(context):
        if expired:
            raise lifecycle.DeadlineReached

    def measure(batches):
        nonlocal expired
        if phase == "error":
            raise RuntimeError()
        for batch in batches:
            visited.append(batch)
            expired = phase == "during"
        expired = phase == "after"
        return len(visited), {"mse": 1.0}

    monkeypatch.setattr(lifecycle, "check_deadline", check)
    result = lifecycle.run_evaluation(
        context=EvaluationContext(
            evaluation_id=uuid4(), output_dir=tmp_path, split="test", model_version="v1"
        ),
        dataloader=[1, 2],
        measure=measure,
        error_code="evaluation_failed",
    )
    assert result.status == ("failed" if phase == "error" else "cancelled")
    assert result.num_examples == 0
    assert result.metrics == {}
    if phase == "before":
        assert visited == []
    elif phase == "during":
        assert visited == [1]
    elif phase == "error":
        assert result.issues[0].message == "RuntimeError"
