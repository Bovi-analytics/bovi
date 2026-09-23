# Repository guide: structure, purpose, and handover

## The model at a glance

The original documentation in <code>bovi-models-template</code> describes a
three-layer model: a generic core, reusable model examples, and separate
experiments. That design remains a useful way to think about the platform. The
current implementation has consolidated the first two layers into the
<code>bovi</code> monorepo and adds an explicit deployment layer.

~~~text
Experiments and learning                         Product and operations
------------------------                         ----------------------
template / Douwe / tutorial  ---- insights --->  bovi monorepo
standalone core / curve      ---- migration -->      |
                                                     +-- bovi-core
                                                     +-- model packages
                                                     +-- central API
                                                     +-- Azure model apps
                                                     +-- dashboard
~~~

The handover rule is therefore simple: use the sibling repositories to
understand the *why*, the examples, and the evolution; use <code>bovi</code> to
determine what is currently built, tested, deployed, or changed.

## 1. <code>bovi</code>: the current product repository

<code>bovi</code> is the monorepo for the running platform. It uses Python 3.12
and manages its Python workspace with <code>uv</code>; the root commands are
<code>just sync</code>, <code>just test</code>, <code>just lint</code>, and the
various <code>just run-*</code> commands.

| Component | Location | Responsibility |
| --- | --- | --- |
| Framework | <code>packages/bovi-core/</code> | Small shared abstractions: configuration, storage, base models, registry, and utilities. Heavy ML frameworks do not belong in this layer. |
| Model packages | <code>packages/models/</code> | <code>lactation-autoencoder</code> (TensorFlow), <code>bovi-yolo</code> (object detection), <code>lactationcurve</code> (classical curves/ICAR), and <code>bestpred</code> (Python port). |
| Central API | <code>apps/backend/api/</code> | One public contract, authentication, SQLite persistence, ingestion, and proxy/orchestration. |
| Model apps | <code>apps/backend/models/</code> | Independently deployable Azure Function Apps for the autoencoder and classical curves. |
| Dashboard | <code>apps/frontend/dashboard/</code> | Next.js interface; browser code reaches the central API through the local <code>/api/bovi</code> proxy. |
| Infrastructure | <code>apps/infrastructure/</code> and <code>packages/infrastructure/pulumi/</code> | Deployment and cloud configuration. |

The operational data flow is:

~~~text
Dashboard -> dashboard proxy (/api/bovi) -> central API
          -> SQLite/storage and internal model Function Apps
~~~

The dashboard boundary is intentional: the dashboard must never call a model
app directly. The central API is the integration boundary.

### Model extensions in the monorepo

<code>bovi-core</code> keeps the plugin registries generic. Concrete model
packages publish provider and predictor entry points
(<code>bovi.model_providers</code> and <code>bovi.predictors</code>), which the
registries discover lazily through package metadata. A provider creates a fresh
runtime model or loads one from an already resolved checkpoint or deployment
artifact. The predictor receives that runtime model through its constructor.
A new model belongs in its own model package, follows this
registry/entry-point convention, and must not pull ML dependencies into the
core.

## 2. <code>bovi-models-template</code>: the design source

This is the best place to learn the intent behind the structure. The
human-oriented documentation is in <code>misc/users/</code>:

| Topic | Starting document |
| --- | --- |
| Navigation | <code>misc/users/overview.md</code> |
| Three-layer architecture | <code>architecture/project_overview.md</code> |
| Directories and responsibilities | <code>architecture/project_structure.md</code> |
| Registry and extensibility | <code>architecture/registry_system.md</code> |
| Collaboration | <code>workflows/collaboration.md</code>, <code>team_workflow.md</code>, <code>git_workflow.md</code> |
| Lactation experiment | <code>experiments/lactation_pipeline.md</code> |

The template shows a standalone experiment repository with
<code>src/models/</code>, <code>data/experiments/</code>, <code>notebooks/</code>,
<code>tests/</code>, and <code>misc/</code>. Its purpose is to start from a clear
project structure, keep domain-specific models and data in that project, and
move only broadly reusable infrastructure into the core.

The template's human guides explain the current provider, configuration and
data-pipeline boundaries. Its standalone source and package configuration remain
historical, including the editable dependency on <code>../bovi-core</code>.
Use the selected current monorepo environment for the updated examples; the
guide update does not make a standalone installation equivalent to that environment.

## 3. <code>bovi-models-douwe</code>: the experiment example

This repository is a concrete derivative of the template. It primarily
documents how a lactation autoencoder and YOLO model were organised in a
standalone experiment repository:

- <code>src/models/lactation/</code> and <code>src/models/yolo/</code> contain
  domain implementations;
- <code>data/experiments/</code> contains configurations, input, and model
  versions;
- <code>notebooks/experiments/</code> contains executable exploration;
- <code>misc/users/</code> contains architecture, pipeline explanations, and
  workflows.

The human guides connect this domain narrative to the current monorepo APIs.
The sibling's implementation remains a historical example, not a deployment
source: its package metadata still selects the standalone core. The notebook
instructions identify the current environment and any model/data prerequisites.

## 4. <code>bovi-models-tutorial</code>: the onboarding repository

This repository teaches the concepts step by step through notebooks: Python
basics, configuration, registry, data sources, transforms, data loaders,
models, lactation end to end, and then Databricks/Unity Catalog. Start with the
root README and numbered notebooks when someone is new to the framework.

The tutorial repository already uses Python 3.12 and obtains
<code>bovi-core</code>, <code>bovi-yolo</code>, and
<code>lactation-autoencoder</code> as subdirectories from the <code>bovi</code>
Git repository. The core lessons teach provider discovery, injected model
construction and explicit data transforms. Use a selected current monorepo
environment for these lessons: the tutorial's dependency declarations do not
pin the local framework changes described here. Cloud and artifact-dependent
lessons have additional prerequisites and should not be treated as CPU smoke tests.
The [Bovi Core package guide](bovi-core-package.md) provides the continuous reading
route alongside the notebooks.

## 5. Standalone predecessors

| Repository | What it preserves | Current treatment |
| --- | --- | --- |
| <code>bovi-core</code> | Earlier core with registry, base classes, and utilities | Historical source; its updated human documentation directs current examples to the monorepo. The standalone implementation and dependency metadata are not migrated. |
| <code>lactation_curve_core</code> | Original lactationcurve package and curve-fitting API | Migrated predecessor. The standalone repository has a May 2026 release history; the current <code>lactationcurve</code> package in <code>bovi</code> is version 1.1.6 and also documents ISLC and BESTPRED methods. |

Keep these repositories for provenance, publication history, authorship, and
comparison. Changes to current product functionality belong in the monorepo
unless explicitly agreed otherwise.

## 6. A compact reading path for a handover

1. This repository guide and the root README of <code>bovi</code>.
2. <code>bovi-models-template/misc/users/overview.md</code> and
   <code>architecture/project_overview.md</code> for the original design.
3. The [Bovi Core package guide](bovi-core-package.md), then the current model
   package or app that matches the person's task.
4. Only then use <code>bovi-models-douwe</code>, tutorial notebooks, or
   standalone repositories for an experiment, a design decision, or historical
   parity.

This keeps the handover readable: one current product source, with the older
repositories as understandable background rather than competing manuals.
