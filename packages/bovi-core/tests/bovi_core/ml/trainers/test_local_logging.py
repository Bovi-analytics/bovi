import asyncio
import json
import os
import stat
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from bovi_core.ml.models import CheckpointReference
from bovi_core.ml.trainers.context import FederatedTrainingContext, TrainingContext
from bovi_core.ml.trainers.local_logging import LocalTrainingResultLogger
from bovi_core.ml.trainers.logging import (
    LogDestinationStatus,
    ResultLogOutcome,
    TrainingResultLogger,
)
from bovi_core.ml.trainers.results import (
    EpochResult,
    TrainingResult,
    TrainingStatus,
    TrainingStopReason,
)
from pydantic import BaseModel


@pytest.fixture
def local_run(tmp_path):
    result = TrainingResult(
        run_id=uuid4(),
        status=TrainingStatus.COMPLETED,
        stop_reason=TrainingStopReason.EARLY_STOPPING,
        started_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
        completed_at=datetime(2026, 9, 2, 12, 5, tzinfo=UTC),
        epochs=(
            EpochResult(epoch=1, metrics={"train/loss": 0.3}),
            EpochResult(epoch=2, metrics={"train/loss": 0.2}),
        ),
        best_epoch=2,
        best_checkpoint=CheckpointReference(
            uri="file:///tmp/best.ckpt", format="pytorch-state-dict"
        ),
    )
    context = TrainingContext(run_id=result.run_id, output_dir=tmp_path, reason="local experiment")
    return context, result


def manifest_path(context):
    return context.output_dir / "training-results" / f"{context.run_id}.json"


def test_persists_context_result_and_explicit_config_snapshot(local_run):
    class SelectedConfig(BaseModel):
        epochs: int = 2
        features: tuple[str, ...] = ("milk", "days")

    context, result = local_run
    metadata = {"dataset": {"name": "training", "records": 3}, "tags": ["cpu", None, True]}
    config = SelectedConfig().model_dump(mode="json")
    logger: TrainingResultLogger = LocalTrainingResultLogger(
        metadata=metadata, config_snapshot=config
    )
    metadata["dataset"]["records"] = 99
    config["epochs"] = 99

    outcome = asyncio.run(logger.log(context, result))

    destination = outcome.destinations[0]
    assert destination.status is LogDestinationStatus.SUCCESS
    assert destination.attempt_count == 1
    assert destination.location == manifest_path(context).as_uri()
    manifest = json.loads(manifest_path(context).read_text())
    assert manifest["schema_version"] == 1
    assert TrainingContext.model_validate(manifest["context"]) == context
    assert TrainingResult.model_validate(manifest["result"]) == result
    assert manifest["metadata"]["dataset"]["records"] == 3
    assert manifest["config_snapshot"] == {"epochs": 2, "features": ["milk", "days"]}
    # The referenced fixture checkpoint does not exist: logging only persists its reference.
    assert list(context.output_dir.iterdir()) == [manifest_path(context).parent]
    assert list(manifest_path(context).parent.iterdir()) == [manifest_path(context)]
    assert ResultLogOutcome.model_validate_json(outcome.model_dump_json()) == outcome


def test_federated_context_fields_are_preserved(local_run):
    context, result = local_run
    federated = FederatedTrainingContext(
        **context.model_dump(),
        experiment_id=1,
        farm_id=2,
        round_id=3,
        attempt=1,
        base_global_model_version="global-3",
    )
    asyncio.run(LocalTrainingResultLogger().log(federated, result))
    manifest = json.loads(manifest_path(context).read_text())
    assert FederatedTrainingContext.model_validate(manifest["context"]) == federated


def test_mismatched_ids_return_issue_without_writing(local_run):
    context, result = local_run
    result = result.model_copy(update={"run_id": uuid4()})
    outcome = asyncio.run(LocalTrainingResultLogger().log(context, result))
    assert outcome.run_id == context.run_id
    destination = outcome.destinations[0]
    assert destination.status is LogDestinationStatus.FAILED
    assert destination.issues[0].code == "logging.run_id_mismatch"
    assert destination.issues[0].write_attempt == 1
    assert not manifest_path(context).exists()


def test_identical_retry_does_not_replace_manifest(local_run):
    context, result = local_run
    first = asyncio.run(LocalTrainingResultLogger(metadata={"b": 2, "a": 1}).log(context, result))
    before = manifest_path(context).stat()
    second = asyncio.run(LocalTrainingResultLogger(metadata={"a": 1, "b": 2}).log(context, result))
    assert second == first
    assert manifest_path(context).stat().st_mtime_ns == before.st_mtime_ns
    assert manifest_path(context).stat().st_ino == before.st_ino


@pytest.mark.parametrize("changed", ["result", "context", "metadata", "config", "corrupt"])
def test_same_run_conflicts_never_overwrite_existing_manifest(local_run, changed):
    context, result = local_run
    asyncio.run(LocalTrainingResultLogger().log(context, result))
    path = manifest_path(context)
    if changed == "corrupt":
        path.write_bytes(b"incomplete previous manifest")
    before = path.read_bytes()
    if changed == "result":
        result = result.model_copy(update={"best_epoch": 1})
    if changed == "context":
        context = context.model_copy(update={"reason": "different"})
    logger = LocalTrainingResultLogger(
        metadata={"source": "different"} if changed == "metadata" else None,
        config_snapshot={"epochs": 3} if changed == "config" else None,
    )
    outcome = asyncio.run(logger.log(context, result))
    assert outcome.destinations[0].status is LogDestinationStatus.FAILED
    assert outcome.destinations[0].issues[0].code == "logging.local_collision"
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_reused_output_directory_keeps_separate_runs(local_run):
    context, result = local_run
    second_result = result.model_copy(update={"run_id": uuid4()})
    second_context = context.model_copy(update={"run_id": second_result.run_id})
    logger = LocalTrainingResultLogger()
    asyncio.run(logger.log(context, result))
    asyncio.run(logger.log(second_context, second_result))
    assert len(list(manifest_path(context).parent.glob("*.json"))) == 2
    assert json.loads(manifest_path(context).read_text())["result"]["run_id"] == str(result.run_id)


@pytest.mark.parametrize("stage", ["mkdir", "fsync", "link"])
def test_write_failures_return_structured_issue_and_leave_no_partial_manifest(
    local_run, monkeypatch, stage
):
    context, result = local_run

    def fail(*args, **kwargs):
        raise OSError()

    if stage == "mkdir":
        monkeypatch.setattr(Path, "mkdir", fail)
    else:
        monkeypatch.setattr(f"bovi_core.ml.trainers.local_logging.os.{stage}", fail)
    outcome = asyncio.run(LocalTrainingResultLogger().log(context, result))
    destination = outcome.destinations[0]
    assert destination.status is LogDestinationStatus.FAILED
    assert destination.location is None
    assert destination.attempt_count == 1
    assert destination.issues[0].message == "OSError"
    assert destination.issues[0].exception_type == "OSError"
    assert destination.issues[0].write_attempt == 1
    assert result.status is TrainingStatus.COMPLETED
    assert not manifest_path(context).exists()
    assert not list(context.output_dir.rglob(".manifest-*"))


@pytest.mark.parametrize("invalid", [object(), float("nan"), float("inf")])
def test_non_json_snapshot_returns_failure(local_run, invalid):
    context, result = local_run
    outcome = asyncio.run(LocalTrainingResultLogger(metadata={"bad": invalid}).log(context, result))
    assert outcome.destinations[0].status is LogDestinationStatus.FAILED
    assert not manifest_path(context).exists()


def test_concurrent_identical_writes_are_idempotent(local_run):
    context, result = local_run

    async def write_concurrently():
        return await asyncio.gather(
            *(LocalTrainingResultLogger().log(context, result) for _ in range(8))
        )

    outcomes = asyncio.run(write_concurrently())
    assert all(
        outcome.destinations[0].status is LogDestinationStatus.SUCCESS for outcome in outcomes
    )
    assert list(manifest_path(context).parent.iterdir()) == [manifest_path(context)]


def test_file_work_runs_off_event_loop_thread(local_run, monkeypatch):
    context, result = local_run
    loop_thread = threading.get_ident()
    worker_threads = []
    publish = LocalTrainingResultLogger._publish

    def record_thread(path, payload):
        worker_threads.append(threading.get_ident())
        publish(path, payload)

    monkeypatch.setattr(LocalTrainingResultLogger, "_publish", staticmethod(record_thread))
    outcome = asyncio.run(LocalTrainingResultLogger().log(context, result))
    assert outcome.destinations[0].status is LogDestinationStatus.SUCCESS
    assert len(worker_threads) == 1
    assert worker_threads[0] != loop_thread


def test_concurrent_conflicting_writers_have_exactly_one_winner(local_run):
    context, result = local_run

    async def write_concurrently():
        return await asyncio.gather(
            *(
                LocalTrainingResultLogger(metadata={"writer": writer}).log(context, result)
                for writer in range(8)
            )
        )

    outcomes = asyncio.run(write_concurrently())
    successful_writers = [
        writer
        for writer, outcome in enumerate(outcomes)
        if outcome.destinations[0].status is LogDestinationStatus.SUCCESS
    ]
    assert len(successful_writers) == 1
    manifest = json.loads(manifest_path(context).read_text())
    assert manifest["metadata"]["writer"] == successful_writers[0]
    for writer, outcome in enumerate(outcomes):
        if writer != successful_writers[0]:
            assert outcome.destinations[0].issues[0].code == "logging.local_collision"
    assert list(manifest_path(context).parent.iterdir()) == [manifest_path(context)]


def test_public_trainers_export():
    from bovi_core.ml.trainers import LocalTrainingResultLogger as ExportedLogger

    assert ExportedLogger is LocalTrainingResultLogger


def test_result_is_snapshotted_before_thread_dispatch(local_run, monkeypatch):
    context, result = local_run
    expected_metrics = dict(result.epochs[0].metrics)
    original_to_thread = asyncio.to_thread

    async def mutate_then_dispatch(function, *args):
        result.epochs[0].metrics["train/loss"] = 999.0
        return await original_to_thread(function, *args)

    monkeypatch.setattr(asyncio, "to_thread", mutate_then_dispatch)
    outcome = asyncio.run(LocalTrainingResultLogger().log(context, result))
    assert outcome.destinations[0].status is LogDestinationStatus.SUCCESS
    manifest = json.loads(manifest_path(context).read_text())
    assert manifest["result"]["epochs"][0]["metrics"] == expected_metrics


@pytest.mark.parametrize("field", ["metadata", "config_snapshot"])
def test_snapshot_copy_error_is_reported_from_log_not_constructor(local_run, field):
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError()

    context, result = local_run
    # Deliberately violate the JSON input contract to exercise the failure path.
    logger = LocalTrainingResultLogger(
        **{field: {"value": CannotCopy()}}  # pyright: ignore[reportArgumentType]
    )
    outcome = asyncio.run(logger.log(context, result))
    destination = outcome.destinations[0]
    assert destination.status is LogDestinationStatus.FAILED
    assert destination.issues[0].code == "logging.snapshot_failed"
    assert destination.issues[0].message == "RuntimeError"
    assert destination.issues[0].write_attempt == 1
    assert not manifest_path(context).exists()


def test_invalid_config_mapping_is_reported_from_log(local_run):
    context, result = local_run
    logger = LocalTrainingResultLogger(config_snapshot=42)  # pyright: ignore[reportArgumentType]
    outcome = asyncio.run(logger.log(context, result))
    assert outcome.destinations[0].status is LogDestinationStatus.FAILED
    assert outcome.destinations[0].issues[0].code == "logging.snapshot_failed"


@pytest.mark.skipif(os.name != "posix", reason="Directory fsync is POSIX-specific")
def test_parent_directory_is_synced_after_link_publication(local_run, monkeypatch):
    context, result = local_run
    events = []
    original_fsync = os.fsync
    original_link = os.link

    def record_fsync(descriptor):
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        events.append("directory_sync" if is_directory else "file_sync")
        if is_directory:
            assert manifest_path(context).exists()
            assert not list(manifest_path(context).parent.glob(".manifest-*"))
        original_fsync(descriptor)

    def record_link(*args, **kwargs):
        events.append("link")
        original_link(*args, **kwargs)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "link", record_link)
    outcome = asyncio.run(LocalTrainingResultLogger().log(context, result))
    assert outcome.destinations[0].status is LogDestinationStatus.SUCCESS
    directories = (manifest_path(context).parent, *manifest_path(context).parent.parents)
    assert events == ["file_sync", "link", *["directory_sync" for _ in directories]]


@pytest.mark.skipif(os.name != "posix", reason="Directory fsync is POSIX-specific")
def test_first_write_syncs_new_ancestors_and_retry_syncs_them_again(local_run, monkeypatch):
    context, result = local_run
    context = context.model_copy(update={"output_dir": context.output_dir / "new" / "nested"})
    original_fsync = os.fsync
    original_open = os.open
    descriptors = {}
    synced = []

    def record_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if flags & os.O_DIRECTORY:
            descriptors[descriptor] = Path(path)
        return descriptor

    def record_sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            synced.append(descriptors[descriptor])
        original_fsync(descriptor)

    monkeypatch.setattr(os, "open", record_open)
    monkeypatch.setattr(os, "fsync", record_sync)
    logger = LocalTrainingResultLogger()
    expected = [manifest_path(context).parent, *manifest_path(context).parent.parents]
    for _ in range(2):
        synced.clear()
        outcome = asyncio.run(logger.log(context, result))
        assert outcome.destinations[0].status is LogDestinationStatus.SUCCESS
        assert synced == expected


@pytest.mark.skipif(os.name != "posix", reason="Directory fsync is POSIX-specific")
def test_directory_sync_failure_is_reported_and_identical_retry_recovers(local_run, monkeypatch):
    context, result = local_run
    original_fsync = os.fsync
    directory_descriptors = []

    def fail_directory_sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_descriptors.append(descriptor)
            raise OSError("directory sync failed")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_sync)
    logger = LocalTrainingResultLogger()
    outcome = asyncio.run(logger.log(context, result))
    assert outcome.destinations[0].status is LogDestinationStatus.FAILED
    assert outcome.destinations[0].issues[0].message == "directory sync failed"
    before = manifest_path(context).read_bytes()
    with pytest.raises(OSError):
        os.fstat(directory_descriptors[0])
    monkeypatch.setattr(os, "fsync", original_fsync)
    retry = asyncio.run(logger.log(context, result))
    assert retry.destinations[0].status is LogDestinationStatus.SUCCESS
    assert manifest_path(context).read_bytes() == before
