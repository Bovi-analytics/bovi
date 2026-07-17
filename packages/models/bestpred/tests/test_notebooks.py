from __future__ import annotations

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks"
EXPECTED_NOTEBOOKS = (
    "00_start_here.ipynb",
    "01_dataframe_quickstart.ipynb",
    "02_legacy_sources_and_cli.ipynb",
    "03_fdd_adapter.ipynb",
    "04_lactationcurve_comparison_and_migration.ipynb",
    "05_legacy_fortran_oracle.ipynb",
)


def test_notebook_series_is_complete_and_ordered() -> None:
    assert tuple(path.name for path in sorted(NOTEBOOK_DIR.glob("*.ipynb"))) == EXPECTED_NOTEBOOKS


@pytest.mark.parametrize("notebook_name", EXPECTED_NOTEBOOKS)
def test_notebook_executes_without_optional_integrations(
    notebook_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BESTPRED_DISABLE_FDD", "1")
    monkeypatch.setenv("BESTPRED_DISABLE_FORTRAN", "1")
    monkeypatch.delenv("BESTPRED_FORTRAN_BINARY", raising=False)

    notebook_path = NOTEBOOK_DIR / notebook_name
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    previous_cwd = Path.cwd()
    try:
        os.chdir(NOTEBOOK_DIR)
        client.execute()
    finally:
        os.chdir(previous_cwd)


def test_committed_notebook_outputs_do_not_expose_local_paths() -> None:
    forbidden = ("/home/", "\\Users\\", ".worktrees/")
    for notebook_name in EXPECTED_NOTEBOOKS:
        notebook = nbformat.read(NOTEBOOK_DIR / notebook_name, as_version=4)
        rendered_outputs = "\n".join(
            str(output)
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
        )
        assert not any(fragment in rendered_outputs for fragment in forbidden)
