# Handover research notes

This is a compact, auditable record of the inventory behind the handover
documentation. It is not an additional user guide.

## Scope

Inventoried repositories: <code>bovi</code>, <code>bovi-core</code>,
<code>bovi-models-douwe</code>, <code>bovi-models-template</code>,
<code>bovi-models-tutorial</code>, and <code>lactation_curve_core</code>.
Together they contain 109 tracked Markdown files; 59 are primary substantive
documentation and 50 are agent instructions, checklists, scratch notes, or
tooling documentation.

## Established facts

- <code>bovi</code> is the current Python 3.12 monorepo. It contains the core,
  four model packages, the central API, two model Function Apps, the dashboard,
  and infrastructure.
- The current core stays lightweight; TensorFlow, PyTorch, and Ultralytics
  belong to model packages, not to <code>packages/bovi-core</code>.
- The dashboard boundary runs through the central API. Browser code uses the
  <code>/api/bovi</code> dashboard proxy, not a direct model-app endpoint.
- The current registry discovers models and predictors through the
  <code>bovi.models</code> and <code>bovi.predictors</code> entry-point groups.
  This replaces the former assumption that a consumer must manually import
  every model package.
- <code>bovi-models-template</code> preserves the useful three-layer intent: a
  generic core, domain-specific models and experiments, and a clear project
  structure.

## Currency review

| Source | Latest relevant documentation | Meaning |
| --- | --- | --- |
| <code>bovi</code> root README | 2026-07-17 | Current operational starting point |
| <code>bovi</code> lactationcurve README | 2026-06-03 | Current package information, version 1.1.6 |
| <code>bovi</code> BESTPRED index | 2026-07-17 | Explicitly separates active and historical material |
| <code>bovi-models-template</code> user docs | 2026-05-08 | Good design source |
| <code>bovi-models-template</code> README/config | 2026-01-15 / 2025-12-17 | Python 3.11 and standalone <code>../bovi-core</code>: do not use as current setup |
| standalone <code>bovi-core</code> README/config | 2025-10-08 | Historical predecessor, Python 3.11 |
| <code>bovi-models-douwe</code> docs | 2026-05-08 | Useful experiment narrative, old editable core link |
| <code>bovi-models-tutorial</code> | 2026-05-08 | Python 3.12 onboarding that already refers to <code>bovi</code> |
| <code>lactation_curve_core</code> | 2026-05-13 | Predecessor of the migrated monorepo package |

## Outdated claims not adopted as instructions

- <code>project_template_v2</code>, a separate <code>bovi-models</code> reference
  package, and placeholders such as <code>yourorg</code> and <code>YourName</code>.
- Python 3.11, standalone <code>uv pip install -e</code> steps, and a direct
  editable dependency on the standalone core.
- Manual imports as mandatory registry discovery. In the monorepo, package
  entry points are the standard; manual registration remains an implementation
  detail when needed.
- A template or standalone experiment repository as a deployment or release
  source.

## Possible next expansion

- Add a short operational runbook per model only if the handover needs more
  depth, beginning with central API contracts.
- Explicitly agree which historical numerical-parity test or publication
  documentation a new owner must know for BESTPRED and lactationcurve.
