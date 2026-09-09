# Bovi Core documentation review: technical follow-up

This is a maintainer reference, not the introductory reading guide. The findings
below came from checking the package documentation against implementation and
running focused local examples. They are not runtime changes made by this review.

Start with [the package guide](bovi-core-package.md) to learn the framework.

## Findings and follow-up work

These are observations from reading and focused local checks, not fixes made by
this documentation change.

| Finding | Evidence and consequence | Suggested next step |
| --- | --- | --- |
| Serving wrappers mishandle dictionaries | In [wrappers.py](../../packages/bovi-core/src/bovi_core/ml/publishing/wrappers.py), `hasattr(input, "values")` also matches a dictionary's method. Local calls reproduce Keras `AttributeError` and PyTorch `TypeError`. | Separate mapping and dataframe handling; test actual exported-model serving parity. |
| Signature conversion is not recursively JSON-safe | [output_to_serializable](../../packages/bovi-core/src/bovi_core/ml/utils/signature_utils.py) leaves arrays inside dictionaries. A local `json.dumps` check fails. | Decide whether the contract is MLflow-compatible values or JSON; implement/test it and relocate beside publishing. |
| Import isolation deserves verification | Image dataset imports Pillow, but [core dependencies](../../packages/bovi-core/pyproject.toml) do not declare it. A populated workspace can hide this. | Clean-install import smoke tests; decide lazy optional import versus a declared dependency. This was not tested in an isolated installation here. |
| Misleading utility placeholders | [data_utils.py](../../packages/bovi-core/src/bovi_core/utils/data_utils.py) describes conversion/benchmark behavior but returns `not_implemented`; `unity_utils.py` is empty. | Remove unused placeholders or implement explicit, tested services; do not present them as existing capabilities. |
| Config-free construction is incomplete | Native loader APIs and prediction interface still depend on legacy `Config`. | Extract explicit runtime settings while preserving a separate YAML construction route. |
| Optional signatures can conceal serving mismatch | Failed output inference may yield an input-only signature; predictor and pyfunc wrapper can accept different forms. | Test the same serving example through the exported wrapper, not only the runtime predictor. |

Full crash recovery, cloud result-log adapters, exact-state resume and federated
aggregation are distinct future capabilities. The
[trainer limitations](trainer-module.md#current-limitations-and-future-work)
remain the detailed record; their presence in a design discussion does not mean
they are already implemented.
