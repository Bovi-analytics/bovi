"""Local, immutable training manifests without framework or cloud dependencies."""

import asyncio
import json
import os
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from uuid import UUID

from pydantic import JsonValue

from .context import TrainingContext
from .issues import Issue
from .logging import LogDestinationResult, LogDestinationStatus, LogIssue, ResultLogOutcome
from .results import TrainingResult


class LocalTrainingResultLogger:
    """Persist context, result and explicitly selected JSON snapshots after training.

    The manifest lives at ``context.output_dir/training-results/<run_id>.json``.
    Identical retries succeed; a different manifest for the same ID fails without
    overwriting the original. Native checkpoint artifacts are referenced, never
    read or copied. Callers must exclude secrets from the supplied snapshots.

    ``metadata`` and ``config_snapshot`` accept JSON-valued mappings, not a whole
    Config instance. Convert selected typed config models with
    ``model_dump(mode="json")`` before passing them. Inputs are copied so later
    caller mutations do not change the captured snapshots.
    """

    def __init__(
        self,
        *,
        metadata: Mapping[str, JsonValue] | None = None,
        config_snapshot: Mapping[str, JsonValue] | None = None,
    ) -> None:
        self._snapshot_issue: Issue | None = None
        self._metadata = {}
        self._config_snapshot = {}
        try:
            self._metadata = deepcopy(dict(metadata)) if metadata is not None else {}
            self._config_snapshot = (
                deepcopy(dict(config_snapshot)) if config_snapshot is not None else {}
            )
        except Exception as exception:
            self._snapshot_issue = Issue.from_exception(exception, code="logging.snapshot_failed")

    async def log(self, context: TrainingContext, result: TrainingResult) -> ResultLogOutcome:
        """Return a destination outcome without changing or failing the training result."""
        if self._snapshot_issue is not None:
            return self._failed_outcome(context.run_id, self._snapshot_issue)
        try:
            # Frozen Pydantic models still contain mutable metric/metadata dicts.
            # Capture them before yielding control to another coroutine or thread.
            context_snapshot = deepcopy(context)
            result_snapshot = deepcopy(result)
            return await asyncio.to_thread(self._log_sync, context_snapshot, result_snapshot)
        except Exception as exception:
            issue = Issue.from_exception(exception, code="logging.snapshot_failed")
            return self._failed_outcome(context.run_id, issue)

    @staticmethod
    def _failed_outcome(run_id: UUID, issue: Issue) -> ResultLogOutcome:
        return ResultLogOutcome(
            run_id=run_id,
            destinations=(
                LogDestinationResult(
                    destination="local",
                    status=LogDestinationStatus.FAILED,
                    attempt_count=1,
                    issues=(LogIssue(**issue.model_dump(), write_attempt=1),),
                ),
            ),
        )

    def _log_sync(self, context: TrainingContext, result: TrainingResult) -> ResultLogOutcome:
        code = "logging.local_write_failed"
        try:
            if context.run_id != result.run_id:
                code = "logging.run_id_mismatch"
                raise ValueError("Training context and result run IDs must match")

            manifest = {
                "schema_version": 1,
                "context": context.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
                "metadata": self._metadata,
                "config_snapshot": self._config_snapshot,
            }
            payload = (
                json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
            ).encode("utf-8")
            path = context.output_dir.absolute() / "training-results" / f"{context.run_id}.json"
            self._publish(path, payload)
            destination = LogDestinationResult(
                destination="local",
                status=LogDestinationStatus.SUCCESS,
                location=path.as_uri(),
                attempt_count=1,
            )
        except Exception as exception:
            if isinstance(exception, FileExistsError):
                code = "logging.local_collision"
            issue = Issue.from_exception(exception, code=code)
            return self._failed_outcome(context.run_id, issue)
        return ResultLogOutcome(run_id=context.run_id, destinations=(destination,))

    @staticmethod
    def _publish(path: Path, payload: bytes) -> None:
        """Publish complete bytes atomically, without a check-then-overwrite race."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".manifest-", delete=False
            ) as f:
                temporary = Path(f.name)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            # A same-directory hard link exposes only complete bytes and cannot
            # replace an existing manifest, even with concurrent writer processes.
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise FileExistsError(f"A different training manifest already exists at {path}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if os.name == "posix":
            # Sync child contents before the parent's directory entry. Include
            # existing ancestors: a retry or concurrent writer may have created
            # them without completing its own durability step.
            for directory in (path.parent, *path.parent.parents):
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
