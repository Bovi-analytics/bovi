# Agent Instructions

Read `docs/bestpred-ai-wiki.md`, `docs/python-port-refactor-plan.md`, `docs/fortran-kernel-trace.md`, and `docs/fortran-quirks.md` before doing any work in this repo. They contain the project goal, repo map, Fortran/Python analysis, build/run status, known mismatch with `DCRexample.results.dcr`, current Python port phase, Fortran kernel trace, next-step strategy, and the list of legacy Fortran quirks we may reproduce temporarily for parity but do not necessarily want to keep in Python.

Key rules:

- Do not run `./bestpred` directly in the repo for experiments; it overwrites fixed output files. Use `scripts/run_bestpred_profiles.py --out-dir /tmp/bestpred-profile-runs`.
- Treat `DCRexample.results.dcr` as legacy reference output, not yet as proven truth for the current source.
- Treat the current Linux Fortran output as the oracle for the Python port; the recovered original macOS binary was run via Darling and produced byte-identical output.
- Keep new work in this `bestpred` repo for now; do not modify sibling `../bovi` unless explicitly asked.
- Use `make -f makefile.gnu` to build. `gfortran` is required.
- New Python port lives in `python/`; validate it with `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run basedpyright` from that directory.
