# Notebook verification

This records the sibling-notebook checks performed on 9 September 2026 against
the current Bovi worktree. It is a verification record, not another introductory
guide. Start with [Understanding Bovi Core](bovi-core-package.md) to learn the
concepts.

## What was checked

There are 32 notebooks across the template, Douwe, tutorial and standalone-core
repositories. One is a reading-only setup lesson. The other 31 were checked in
real Python 3.12 Jupyter kernels using the current monorepo environment.

The tables deliberately distinguish complete local lessons from mixed lessons
whose local work passed but whose cloud operations were not executed.
Running a cell that prints a disabled-cloud message does not test its cloud body.

Existing local YOLO weights, images, lactation data and SavedModels were used
for the model examples. Credentials were not printed, model weights were not
changed, and no remote registration, alias change or secret administration was
performed. Secret-mechanism tests used synthetic local values.

## How to reproduce the checks

Select a monorepo checkout containing the current provider and explicit-transform
interfaces, run its `just sync`, and use that environment as the notebook kernel.
The sibling READMEs explain how to select the current checkout rather than their
historical standalone dependencies. Restart the kernel between lessons: each
notebook was checked independently, not against variables left by another lesson.

Run local lessons with CPU settings. For artifact-dependent lessons, use their
documented environment variables to select existing trusted files.
Do not enable cloud branches merely to make every cell print a success message;
those branches need the relevant credentials, resources and permissions.

The checked kernels used fatal execution errors, not `allow_errors=True`.
For cloud-only workflows, explicit missing-prerequisite errors were recorded as
blocked. Mixed notebooks use visible disabled branches so local sections can
complete without remote access.

## Template repository

Paths below are under `bovi-models-template/notebooks/experiments/`.

| Notebook | Verified | Not verified |
| --- | --- | --- |
| `tutorial/custom_transforms_tutorial.ipynb` | Complete local lesson | None |
| `tutorial/dataloader_tutorial.ipynb` | Complete local lesson | None |
| `tutorial/config_system_tutorial.ipynb` | Local configuration lesson | Real secrets and Blob authentication |
| `tutorial/model_registry_tutorial.ipynb` | Original classifier, registry and local prediction lesson | Unity Catalog publication/reload |
| `tutorial/mock_experiment_local.ipynb` | YOLO inference, filtering, visualization and box formats | Comparison with a different second checkpoint |
| `lactation_autoencoder/lactation_autoencoder_refactored.ipynb` | Real SavedModel loading, local data pipeline, inference, plots and local MLflow serving roundtrip | Remote publication and reload |
| `tutorial/unity_catalog_move_weights_to_unity_catalog.ipynb` | Local setup, loading and naming | Catalog access, publication and remote reload |

The second-model exercise remains in the notebook. Supply
`YOLO_SECOND_WEIGHTS` to exercise it; the available nano checkpoint was not
passed off as a different model.

## Douwe repository

Paths below are under `bovi-models-douwe/notebooks/experiments/`.

| Notebook | Verified | Not verified |
| --- | --- | --- |
| `tutorial/custom_transforms_tutorial.ipynb` | Complete local lesson | None |
| `tutorial/dataloader_tutorial.ipynb` | Complete local lesson | None |
| `tutorial/config_system_tutorial.ipynb` | Local configuration lesson | Real secrets and Blob authentication |
| `tutorial/model_registry_tutorial.ipynb` | Original classifier, registry and local prediction lesson | Unity Catalog publication/reload |
| `tutorial/mock_experiment_local.ipynb` | YOLO inference, filtering, visualization and box formats | Different second checkpoint, via `YOLO_SECOND_WEIGHTS` |
| `lactation_autoencoder/lactation_autoencoder_refactored.ipynb` | Real SavedModel, local data pipeline, raw/rich predictions, plots and local MLflow serving roundtrip | Remote persistence, publication and reload |
| `tutorial/unity_catalog_move_weights_to_unity_catalog.ipynb` | Local setup, loading and naming | Catalog access, publication and remote reload |
| `yolo/yolo_cow_detection.ipynb` | Local detection workflow and explicit preprocessing | Cloud reference examples |

The two lactation notebooks retain their distinct lesson structures. The extra
Douwe sections were not discarded to make the files identical.

## Tutorial repository

Paths below are under `bovi-models-tutorial/notebooks/`.

| Notebook | Verified | Not verified |
| --- | --- | --- |
| `00-setup/00_setting_up_vscode.ipynb` | Inspected: reading-only, no executable cells | Manual editor setup |
| `01-python-fundamentals/01_classes_abstract_protocols.ipynb` | Complete local lesson | None |
| `02-bovi-core/01_config_system.ipynb` | Local lesson with copied fixtures and dummy credentials | Optional cloud-client branch and real credentials |
| `02-bovi-core/02_model_registry.ipynb` | Complete local lesson, including installed-plugin discovery | None |
| `03-data-pipeline/02_custom_transforms.ipynb` | Complete local lesson, including YAML construction | None |
| `03-data-pipeline/03_dataloaders.ipynb` | Complete local lesson, YAML changes and train-only augmentation assertions | None |
| `04-models/02_mock_experiment_local.ipynb` | Real YOLO inference, filtering, visualization and box formats | Different second checkpoint, via `BOVI_YOLO_SECOND_WEIGHTS` |
| `05-end-to-end/01_lactation_autoencoder_pipeline.ipynb` | Real local JSON/statistics, SavedModel inference, plots, serving parity and local MLflow save/load parity | Cloud publication and remote load |
| `06-databricks/01_working_with_unity_catalog.ipynb` | Real images/artifact, prediction, signature, local serving and MLflow save/load parity | Catalog listing, publication and remote load/inference |
| `06-databricks/02_secrets_and_keys.ipynb` | Safe local setup and disabled branches | Real scope listing and secret retrieval |
| `06-databricks/03_model_registration_and_versioning.ipynb` | Local naming, provider/artifact loading, prediction, serving example and signature | Registry version lookup, publication and alias/version loads |

Shared images in the loader exercise illustrate configuration differences.
They are explicitly not presented as a genuine held-out evaluation split.

## Standalone core repository

Paths below are under `bovi-core/notebooks/`. These lessons use the current
monorepo kernel, not the standalone repository's historical implementation.

| Notebook | Verified | Not verified |
| --- | --- | --- |
| `notebook_tests/test_config_and_model_loading.ipynb` | Complete local workflow: configuration, real pretrained artifact, prediction and error handling | None |
| `notebook_tests/secrets_test.ipynb` | Complete synthetic local secret-lookup test | Real secret-store integration |
| `notebook_tests/end_to_end_system_test.ipynb` | Local pretrained prediction, data preparation and configuration checks | Blob access |
| `pipelines/configuration/unity_catalog_move_weights_to_unity_catalog.ipynb` | Local artifact loading, serving parity, signature and naming | Catalog access, registration, aliases and remote reload |
| `notebook_tests/unity_catalog_check_available_models.ipynb` | Local setup only | Cloud workflow blocked by explicit prerequisites |
| `pipelines/configuration/databricks_secrets_add_keys.ipynb` | Local setup only | Cloud workflow blocked by explicit prerequisites |

The standalone source and historical package tests were not migrated. A local
secret lookup does not establish permission to access Databricks secret scopes.

## What was repaired

The changes preserve the teaching subjects rather than replacing difficult
examples with unrelated ones. In particular, the tutorial's YOLO lesson again
teaches detection; its original configuration, visualization and coordinate-format
exercises remain.

The main compatibility repairs were explicit provider/model/predictor construction,
separating transforms from dataset construction, reading selected YAML settings,
and using the underlying image dataset for its inspection helpers.
Installed-plugin discovery is actually exercised rather than simulated with an
already registered notebook class.

Execution also exposed concrete errors that were fixed and rerun:

- Image-dataset inspection methods were called on `TransformedDataset`.
- Albumentations was invoked as a positional sample transform rather than through
  its explicit wrapper.
- Examples referred to configuration path/template attributes unavailable after
  configuration processing.
- Publication naming assumed `config.project.version` existed for a dynamically
  versioned package; it now uses installed distribution metadata.
- A loaded TensorFlow serving wrapper could not be pickled for MLflow.
  Saving a fresh wrapper, which loads its own state later, passed save/load parity.

The local model and serving tests establish compatibility, not model quality.
They also do not validate the unexecuted remote operations.

## Preservation and evidence

The template and Douwe notebook cell counts, cell types and lesson headings/order
were retained. The standalone notebooks retain all 54 original cells and all
19 original Markdown cells. The tutorial's explanations of classes, protocols,
registry discovery and configuration were restored where an initial adaptation
had removed useful teaching content. Its advanced chapters retain their original
section order, data inspection, transforms, plots and publication topics.

Stale outputs and execution counts were cleared from edited source notebooks.
Executed outputs and first-failure evidence were kept separately under `/tmp`,
not written over the curated originals. The initial documentation pass left the
monorepo's hand-curated lactation notebook unchanged. The serving follow-up below
updates only two Markdown reference cells in that notebook; its executable
cells, section order and analysis remain unchanged.

Local verification evidence for this session is in:

- `/tmp/sibling-notebooks-executed/` and `/tmp/notebook-execution-status.json`
  for template/Douwe; earlier failures are in `/tmp/sibling-notebooks-first-pass/`.
- `/tmp/tutorial-restoration.68pMVP/executed/` for the six basic tutorial lessons.
- `/tmp/executed-01_lactation_autoencoder_pipeline.ipynb`,
  `/tmp/executed-01_working_with_unity_catalog.ipynb`,
  `/tmp/executed-02_secrets_and_keys.ipynb` and
  `/tmp/executed-03_model_registration_and_versioning.ipynb`
  for the advanced tutorial lessons.
- `/tmp/standalone-core-notebooks-evidence/jupyter/` for standalone-core checks.

These are temporary local evidence paths, not prerequisites or permanent
published artifacts. Checks against staging were followed by patch-replay and
applied-file comparisons. Final prose-only corrections do not imply a cloud
body was executed.

A credential-like literal was removed from a historical notebook. If it was
real, revoke or rotate it; clearing a notebook does not remove its Git history.

No full package regression suite was run for the initial documentation/notebook edits.
The relevant checks were notebook execution, source-preservation comparisons,
syntax, links and Mermaid parsing. Shared runtime code was not changed in this
documentation pass.

## Lactation serving follow-up

The template and Douwe lactation notebooks were rerun in fresh Python 3.12
kernels after correcting the serving contract. Both completed their local
workflows, including a new check against their real, existing SavedModels:

1. Predict from a dictionary of batched arrays through `LactationSavedModelWrapper`.
2. Compare the named outputs with the existing lactation predictor.
3. Infer an MLflow tensor signature from those serving inputs and outputs.
4. Save a fresh wrapper and the artifact into a temporary local MLflow model.
5. Reload it with `mlflow.pyfunc.load_model` and verify matching predictions.

Only the serving code cell and its introduction changed in each sibling
notebook. Executed copies are in `/tmp/lactation-serving-verification/`, named
`bovi-models-template.ipynb` and `bovi-models-douwe.ipynb`. The source notebooks
retain their original cell order and contain no saved execution outputs.

Eight focused tests in `test_lactation_publishing.py` also passed. These use a
small real TensorFlow SavedModel to cover batch sizes one and two, integer
signature inputs, local MLflow save/load, input preservation, and rejection of
missing, extra, unbatched, empty or inconsistent inputs.

The follow-up also passed `just test`: 1,520 tests passed, 30 were skipped and
96 were deselected across its four test groups. Ruff passed for the changed
Python files, and basedpyright reported no errors or warnings for the two
serving modules. Sandbox-restricted Jupyter startup and the first API test
stalled, so the notebook and regression runs were repeated with approved local
execution outside the sandbox. Cloud branches stayed disabled.

Cloud publication and remote reload were not executed. The local roundtrip uses
the installed worktree environment: it does not prove that a clean serving
environment can install the recorded Bovi package versions. Those packages must
be made available to the deployment environment before remote publication.
