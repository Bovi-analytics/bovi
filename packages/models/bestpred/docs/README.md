# BESTPRED documentation index

The active user guide is the package [`README.md`](../README.md). The material
below is preserved from the standalone BESTPRED repository so that numerical
decisions and Fortran parity remain auditable.

## Maintained package documentation

- [`../README.md`](../README.md): installation, APIs, inputs, units, outputs,
  CLI use, and Bovi comparison.
- [`reference/port/bovi_fdd_alignment.md`](reference/port/bovi_fdd_alignment.md):
  current FDD boundary and missing domain models.
- [`reference/port/fortran-quirks.md`](reference/port/fortran-quirks.md): known
  compatibility behavior that must not be mistaken for desired new behavior.

## Historical port snapshots

These files are copied without rewriting their historical paths or status
claims:

- [`reference/port/bestpred-ai-wiki.md`](reference/port/bestpred-ai-wiki.md)
- [`reference/port/fortran-kernel-trace.md`](reference/port/fortran-kernel-trace.md)
- [`reference/port/python-port-refactor-plan.md`](reference/port/python-port-refactor-plan.md)
- [`reference/legacy/python-package-README.md`](reference/legacy/python-package-README.md)
- [`reference/legacy/AGENTS.original.md`](reference/legacy/AGENTS.original.md)
- [`reference/legacy/README`](reference/legacy/README)
- [`reference/legacy/CHANGEFILE.txt`](reference/legacy/CHANGEFILE.txt)
- [`reference/legacy/FILELIST.txt`](reference/legacy/FILELIST.txt)

## Visual reports and manual

- [`reference/reports/bestpred-integration-map.html`](reference/reports/bestpred-integration-map.html)
- [`reference/reports/bestpred_fortran_analysis.html`](reference/reports/bestpred_fortran_analysis.html)
- [`reference/reports/bovi-bestpred-comparison.html`](reference/reports/bovi-bestpred-comparison.html)
- [`reference/manual/Best Prediction Manual.pdf`](reference/manual/Best%20Prediction%20Manual.pdf)

## Source snapshot

The imported code is based on standalone BESTPRED commit `454c81d` and includes
the working-tree documentation updates present during migration on 2026-07-17,
including `bestpred-integration-map.html`. Generated environments, caches,
compiled binaries, build files, and transient root outputs were intentionally
excluded. Golden fixtures remain under `tests/fixtures`.
