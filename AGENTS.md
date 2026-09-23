# AGENTS.md

## Project

bovi is a monorepo for the Bovi dairy analytics platform.

Key areas:

- `packages/bovi-core/`: slim ML framework and shared utilities.
- `packages/models/lactation-autoencoder/`: TensorFlow lactation prediction model.
- `packages/models/lactationcurve/`: classical lactation curve fitting models.
- `packages/models/bovi-yolo/`: YOLO object detection for dairy applications.
- `apps/backend/api/`: central FastAPI gateway.
- `apps/backend/models/`: deployable backend model apps.
- `apps/frontend/dashboard/`: Next.js dashboard.

## Required Workflow

Use the user-level `$worktree-first` skill for new implementation work. Create worktrees under `.worktrees/` so parallel work does not interfere with the main checkout.

Before inspecting or editing for implementation work, use the user-level
`$tmux-shell` skill. Choose a concise task name, rename the tmux session and
window with that name, enable terminal-title propagation, and rename the Codex
thread with the same task name using the skill's SQLite command. Do not rely on
`/rename`; Codex cannot execute TUI slash commands itself. Verify the tmux
display message and the Codex thread title query before continuing.

After entering the Bovi worktree, update the tmux `@codex_context` header with
the branch, worktree name, and base commit. Verify the value before running repo
commands so the header reflects Codex's actual worktree, not just the shell's
original directory.

Run `just sync` from the worktree immediately after entering it and before any
tests, linting, typechecking, commit hooks, or import-dependent debugging. This
is mandatory for every new or reused worktree. If a command fails because the
worktree was not synced, treat that as setup error: run `just sync`, then rerun
the failed command.

Use the user-level `$write-tests` skill whenever adding functionality, fixing bugs, refactoring behavior, or touching code with existing tests.

Commit cadence:

- For many tasks or broad changes, make intermittent commits at coherent, tested checkpoints.
- For small focused tasks, one commit is fine.
- Do not commit unrelated user changes.

## Tests

The final regression command for this repo is:

```bash
just test
```

Run targeted tests while developing when useful, but run `just test` before committing unless the environment prevents it. If it cannot be run, report the exact reason and the closest test command that was run.

`just test` is the required final regression command and currently delegates to
the affected-test runner. Use `just test-all` only when an explicit full pytest
run is needed. Do not replace the required `just test` with package-local tests
unless the environment prevents the repo command from running.

When adding functionality:

- Reuse existing `conftest.py` files and fixtures first.
- Add new fixtures only when existing fixtures do not fit.
- Update existing tests first when touched code changes existing behavior.
- Add new tests after checking and updating old tests.
- Add integration tests when behavior crosses package, API, database, service, or app boundaries.

## Commands

From repo root:

```bash
just sync
just test
just lint
just run-api
just run-dashboard
```

Per-package examples:

```bash
cd packages/bovi-core && just test
cd packages/models/bovi-yolo && just test
cd packages/models/lactationcurve && just test
cd apps/backend/api && just test
cd apps/backend/models/lactation-curves && just test
```

## Constraints

- Python 3.12 only.
- Use `uv` for Python dependency management.S
- Use `bun` for the dashboard.
- Import from `bovi_core`, never `src.bovi_core`.
- Register model-family providers with `@ModelProviderRegistry.register("name")` or
  the equivalent explicit decorator call, and publish them through the
  `bovi.model_providers` entry-point group.
- Do not commit model weights.
- Keep `bovi-core` slim; do not add ML framework dependencies there.
- Dashboard must talk to the central API only, never directly to model apps.

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
