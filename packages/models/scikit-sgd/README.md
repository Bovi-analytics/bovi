# scikit-sgd

Incremental native `SGDRegressor` training through the Bovi model and trainer contracts.

## Checkpoints

Core's `LocalCheckpointStore` writes native joblib data into a private staging directory, then publishes an immutable version directory. The manifest covers every payload file with SHA-256 checksums and records the model config, run ID, completed epoch, and `weights_only` recovery scope. Reusing an output directory cannot overwrite an older result reference. Resolve a checkpoint with `bovi_core.ml.models.checkpoints.LocalCheckpointResolver().resolve(reference)`, then pass it to `provider.restore_checkpoint(model_config, resolved)`; do not construct `last.joblib` paths.

Restoration returns a `Model` for a new training attempt, not an exact training resume. Native estimator counters survive joblib serialization, but run bookkeeping, loader position, RNG state, and early-stopping history are not restored. Epoch numbering restarts. A save failure retains completed epoch metrics and prior valid checkpoints. Joblib can execute code when loaded: checksums detect damage, not trustworthiness; only load trusted bundles.

Run `just sync` at the repository root before testing, then `uv run pytest --import-mode=importlib packages/models/scikit-sgd/tests -q`.
