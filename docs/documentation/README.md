# Bovi — Handover and documentation map

This directory is the short starting point for the Bovi handover. It identifies
which repository is the operational source of truth today, which repositories
mainly hold design, experiment, or learning material, and where to find more
detail.

## Start here

1. Read the [Way of working](way-of-working.md) for the contribution,
   deployment, release, and ownership workflow.
2. Read the [Repository guide](repository-guide.md) for the relationships,
   ownership, and current status of every relevant repository.
3. Open the editable [repository structure diagram](repository-structure.drawio)
   for the visual overview of those relationships.
4. Read the root [README](../../README.md) and
   [CLAUDE.md](../../CLAUDE.md) in <code>bovi</code> for the current monorepo,
   local commands, and architecture rules.
5. Then choose the package or application documentation that matches the task
   at hand.

The short version is:

~~~text
                         bovi (current product and integration repository)
                                      |
      +-------------------------------+------------------------------+
      |                               |                              |
  shared core                    model packages                   deployables
  packages/bovi-core             packages/models/*               apps/*
      |                               |                              |
  contracts, registry,       autoencoder, YOLO,              central API,
  configuration, storage     lactationcurve, BESTPRED         model apps, dashboard

Historical/supporting repositories: bovi-models-template, bovi-models-douwe,
bovi-models-tutorial, standalone bovi-core, and lactation_curve_core.
~~~

## Sources and reliability

The dates below are the latest substantive changes to the relevant documentation
or repository, checked on 28 August 2026. A recent date does not guarantee that
every implementation detail is still correct; the monorepo and its
<code>pyproject.toml</code> files are authoritative for current behaviour.

| Source | Role in the handover | Treat as |
| --- | --- | --- |
| [<code>bovi</code>](../../README.md) | Product, packages, API, model apps, and dashboard | **Current source of truth** |
| <code>../bovi-models-template</code> | Original three-layer design and project structure | Design and template source |
| <code>../bovi-models-douwe</code> | Working example of lactation and YOLO experiments | Historical experiment source |
| <code>../bovi-models-tutorial</code> | Framework learning path and notebooks | Onboarding source |
| <code>../bovi-core</code> | Earlier standalone core | Historical predecessor |
| <code>../lactation_curve_core</code> | Earlier standalone lactationcurve package/API | Migrated predecessor |

The last five repositories are not labelled deleted or unusable. For this
handover, they simply are not the source for operational versions, deployment,
or current dependencies.

## Current documentation in <code>bovi</code>

| Topic | Starting point | Use it when |
| --- | --- | --- |
| Contribution and operations | [Way of working](way-of-working.md) | Branches, pull requests, CI, deployment, releases, or ownership |
| Monorepo and local work | [root README](../../README.md), [CLAUDE.md](../../CLAUDE.md) | Always |
| Shared ML framework | <code>packages/bovi-core/</code> and its package configuration | Working on the registry, configuration, storage, or model foundations |
| Classical curves and ICAR | [lactationcurve README](../../packages/models/lactationcurve/README.md) | Curve fitting, LCCs, or 305-day yield |
| BESTPRED port | [BESTPRED documentation index](../../packages/models/bestpred/docs/README.md) | Fortran parity, FDD, or best prediction |
| Dashboard | [dashboard README](../../apps/frontend/dashboard/README.md) | Local UI work or API proxy behaviour |
| API and model apps | <code>apps/backend/api/</code> and <code>apps/backend/models/</code> | Contracts, storage, or deployment |
| 2026 decisions and plans | <code>docs/superpowers/{plans,specs}/</code> | Historical design context, not as a runtime guide |

## Maintenance agreement

- Keep this directory at overview level; do not duplicate package APIs or
  implementation details.
- Document a structural change in the current package or app first, then update
  the repository guide when the relationship between repositories changes.
- Explicitly label older material as *historical* or *template*. Do not remove
  it merely because it is no longer executable: it explains why the split and
  consolidation happened.
- Use the current <code>bovi</code> workspace with Python 3.12, <code>uv</code>,
  <code>just sync</code>, and <code>just test</code>; the older sibling
  repositories still contain Python 3.11 and standalone-installation
  instructions.
