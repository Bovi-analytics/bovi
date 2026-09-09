# pytorch-linear

CPU-only Bovi trainer reference: one linear layer learns `y = 2x + 1` from eight training records and four validation records. No downloads, GPU or external services are needed after installing dependencies.

From the repository root:

```bash
just sync
uv run pytest --import-mode=importlib packages/models/pytorch-linear/tests -q
uv run jupyter nbconvert --to notebook --execute --inplace packages/models/pytorch-linear/notebooks/experiments/pytorch_linear/pytorch_linear_training.ipynb
```

The package follows `dataloaders/{source,dataset,factory}.py`, `models/{config,model,provider}.py`, and `trainers/{config,trainer,evaluator,arrays}.py`. It reuses the core NumPy batcher, then converts batches at the native training boundary. Model providers register through `bovi.model_providers`.

Configs can be constructed directly or through `from_config(Config(...))`. YAML uses `models.pytorch_linear.{architecture,dataset,dataloaders,training,evaluation}`. Training returns every completed epoch's train/validation MSE and MAE, best/last checkpoint references, and structured failures. Deadlines are checked between batches. Early stopping monitors validation MSE, falling back to training MSE. Without a context, no files are written.

Checkpoints use CPU state dictionaries loaded with weights_only=True. SGD has no momentum or learning-rate schedule. Resume restarts epoch numbering; it does not restore early-stopping history or arbitrary dataloader shuffle state. Use a separate output directory per attempt. The supplied non-shuffled example tests split-run equivalence. Models start with zero weights for a deterministic tutorial, not as a general neural-network initialization policy.

The notebook uses a temporary output directory and removes its generated checkpoints. Neither example is a production farm prediction model.
