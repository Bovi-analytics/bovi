# Local Training Result Logging

Call the local logger from the orchestrator or notebook after training. It does
not run inside a trainer and never changes the returned `TrainingResult`.

```python
from bovi_core.ml.trainers import LocalTrainingResultLogger

logger = LocalTrainingResultLogger(
    metadata={"dataset": {"name": "train", "records": len(train_loader.dataset)}},
    config_snapshot={
        "model": model_config.model_dump(mode="json"),
        "training": training_config.model_dump(mode="json"),
    },
)
log_outcome = await logger.log(context, result)
log_outcome.model_dump(mode="json")
```

Supply only explicitly selected JSON-compatible settings and metadata. Never pass
the whole `Config`, secrets, clients, or credentials. The constructor copies the
snapshots; subsequent mutations of the original mappings do not alter the log.
Non-JSON values and nonfinite numbers produce a structured failed logging outcome.

Each manifest is stored at
`context.output_dir/training-results/<context.run_id>.json`. Its versioned JSON
contains `context`, `result`, `metadata`, and `config_snapshot`. Federated context
fields are retained. Checkpoint references are recorded, but native artifacts
are not read, copied, validated, or exported. Keep those artifacts separately.

Run IDs must match between context and result. Reusing an output directory with
different run IDs keeps separate manifests. An identical retry succeeds without
replacing the existing file. Changed context, result, or snapshots for an existing
run ID fail with `logging.local_collision`; this is an immutable final-result log,
not an append-only epoch stream. Corrupt existing files are not overwritten.

Context and result are deep-copied when the `log()` coroutine begins executing,
before its first await. Later mutations to nested metrics cannot change that
attempt's manifest. Creating a coroutine without awaiting or scheduling it does
not capture a snapshot. Snapshot preparation errors are returned as structured
failures, including invalid constructor mappings that cannot be copied.
JSON serialization and file operations run in `asyncio.to_thread`. Every call makes one
logging attempt, with errors reported through `ResultLogOutcome`,
`LogDestinationResult`, and `LogIssue(write_attempt=1)`. Inspect the outcome
separately from training success. Cancellation propagates normally; cancellation
of the awaiting coroutine does not stop an already-running filesystem thread.

Publication uses a flushed/fsynced temporary file and an atomic, non-overwriting
hard link in the same directory. Concurrent identical writers are safe. This
requires a trusted local filesystem with hard-link support; unsupported filesystems
return a failed logging outcome. After publication and temporary-file cleanup,
the manifest's directory and all ancestors up to the filesystem root are fsynced
in child-to-parent order on POSIX, including on identical retries. This covers
newly created output-directory ancestors and concurrent first writers.
A directory sync failure
returns failure even though the complete manifest may already be present; an
identical retry can complete the sync. Other platforms do not provide this
directory durability step. The logger does not provide cloud synchronization or automatic retries. Temporary
files are removed on normal success/failure, but process termination can leave
hidden `.manifest-*` files. Azure/BlobStore and MLflow adapters are future work.
