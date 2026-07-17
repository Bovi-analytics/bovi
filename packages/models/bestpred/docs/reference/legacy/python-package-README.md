# bestpred-py

Python port of the BESTPRED lactation best-prediction model.

The current Fortran/macOS BESTPRED output is the oracle for this package. The
distributed `DCRexample.results.dcr` file is retained as a legacy/manual
reference because it does not match the current source and recovered macOS
binary.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
```

## CLI

```bash
uv run bestpred run \
  --source 11 \
  --input tests/fixtures/source11_current/DCRexample.txt \
  --par tests/fixtures/source11_current/bestpred.par \
  --output /tmp/results_v2.dcr
```

Source 11 parsing and simulation scaffolding is implemented first. The full
numerical BESTPRED kernel is ported incrementally behind these typed interfaces.
