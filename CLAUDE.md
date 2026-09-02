# CLAUDE.md

## Project

bovi is a monorepo for the Bovi dairy analytics platform containing:

### Packages (libraries)
- **packages/bovi-core/** — Slim ML framework: base classes, registries, config, utilities (published to PyPI)
- **packages/models/lactation-autoencoder/** — TensorFlow autoencoder for milk production prediction
- **packages/models/lactationcurve/** — Classical lactation curve fitting: Wood, MilkBot, Wilmink, etc. (published to PyPI)
- **packages/models/bovi-yolo/** — YOLO object detection for dairy applications
- **packages/infrastructure/pulumi/** — Azure infrastructure as code

### Apps (deployables)
- **apps/backend/api/** — Central FastAPI gateway: unified contract, SQLite persistence, monitoring
- **apps/backend/models/lactation-curves/** — Azure Function App: classical curve fitting + milkbot
- **apps/backend/models/lactation-autoencoder/** — Azure Function App: TF autoencoder predictions
- **apps/frontend/dashboard/** — Next.js visualization dashboard (bun)

### Data flow
Dashboard → Central API → SQLite on Azure Files + proxies to model Function Apps (internal)

## Commands

### Workspace (from repo root)
```bash
just sync                        # Install all workspace dependencies
just test                        # Run tests affected by the current Git changes
just test-all                    # Run the full Python test suite
just lint                        # Lint and format all code
just run-api                     # Run central API locally
just run-dashboard               # Run dashboard locally (bun)
```

### Per-package (cd into the package directory first)
```bash
cd packages/bovi-core && just test
cd packages/models/bovi-yolo && just test
cd packages/models/lactationcurve && just test
cd packages/models/lactationcurve && just build
cd packages/models/lactationcurve && just publish
cd apps/backend/api && just test
cd apps/backend/models/lactation-curves && just test
```

## Architecture

```
External repos (depend on bovi-core via PyPI):
  bovi-models-template    — Minimal skeleton for new users
  bovi-models-example     — Template with worked examples
  bovi-private            — Private models (separate private repo)
```

## Critical Rules

- Python 3.12 only
- Import from `bovi_core`, never `src.bovi_core` (breaks singletons)
- Register models with `@ModelRegistry.register("name")`
- Model weights stored in Azure Blob Storage, never committed to git
- Follow PEP8, use ruff for formatting, basedpyright for type checking
- Use `uv` as Python package manager, `bun` for frontend
- bovi-core must stay slim — no ML framework deps (torch, tf, etc.)
- Each backend model app is independently deployable
- Dashboard talks to central API only, never directly to model apps

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **bovi** (11176 symbols, 19766 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/bovi/context` | Codebase overview, check index freshness |
| `gitnexus://repo/bovi/clusters` | All functional areas |
| `gitnexus://repo/bovi/processes` | All execution flows |
| `gitnexus://repo/bovi/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
