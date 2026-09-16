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

This project is indexed by GitNexus as **bovi** (8445 symbols, 16072 relationships, 496 execution flows).

> Index stale? Run `node .gitnexus/run.cjs analyze --index-only` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? Bootstrap with `npx`, `bunx`, or `pnpm dlx` — e.g. `bunx gitnexus@latest analyze` (npm 11 npx crash; #1939).

## Always Do

- **MUST run impact before editing.** Use `impact({target: "symbolName", direction: "upstream"})` or `node .gitnexus/run.cjs impact "symbolName" --direction upstream --repo .`; report callers, processes, and risk. Never substitute grep for graph analysis.
- **MUST analyze graph changes before committing.** Use `detect_changes({scope: "all"})` (MCP) or `node .gitnexus/run.cjs detect-changes --scope all --repo .` (CLI fallback). `partial: true` or `truncated: true` is not a clean check — a zero means unseen, not unaffected; re-run it. For regression review: `detect_changes({scope: "compare", base_ref: "main"})` or `node .gitnexus/run.cjs detect-changes --scope compare --base-ref "main" --repo .`.
- MUST warn on HIGH/CRITICAL `risk` pre-edit; never use `riskSharedAxes` to waive a HIGH/CRITICAL `risk` warning. Compare File/symbol: MCP File omits axes; Graph-RAG expands File.
- **MUST treat `risk: UNKNOWN` as unresolved, not as low.** An empty caller set is not evidence the symbol is unused — it can also mean the callers are not resolvable by the index (plain-object property access, dynamic dispatch, cross-language calls). `impact` pairs `UNKNOWN` with a `riskNote` saying so. Confirm with a text search before treating the symbol as safe to change or delete; do not proceed on the strength of a zero.
- **MUST use `query({search_query: "concept"})` for concepts/flows, `context({name: "symbolName"})` for a named symbol, or `impact` for blast radius, on read-only callers, dependencies, imports, or execution flow.** Graph first; text search only for empty/`UNKNOWN`/literals.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method before MCP/CLI impact analysis.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis, and never read `UNKNOWN` as an all-clear — it means the walk could not answer, which is the one verdict that requires confirming by other means.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit before MCP/CLI graph change analysis.

## Resources

| Resource | Use for |
| --- | --- |
| `gitnexus://repo/bovi/context` | Codebase overview, check index freshness |
| `gitnexus://repo/bovi/clusters` | All functional areas |
| `gitnexus://repo/bovi/processes` | All execution flows |
| `gitnexus://repo/bovi/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
| --- | --- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
