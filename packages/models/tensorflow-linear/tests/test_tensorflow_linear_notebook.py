"""Execute the public CPU tutorial in a fresh kernel."""

import json
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

pytestmark = pytest.mark.tensorflow

NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks/experiments/tensorflow_linear/tensorflow_linear_training.ipynb"
)


def test_tensorflow_linear_tutorial_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("BOVI_NOTEBOOK_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    executed = client.execute()
    nbformat.write(executed, tmp_path / "executed.ipynb")

    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    plots = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    assert len(plots) >= 2, "The tutorial must show learning and validation plots"
    manifests = [
        json.loads(path.read_text())
        for path in tmp_path.glob("bovi-tensorflow-linear/*/*/training-results/*.json")
    ]
    assert len(manifests) == 2
    assert all(run["result"]["status"] == "completed" for run in manifests)
    first = next(run for run in manifests if not run["context"]["resumed_from_run_id"])
    resumed = next(run for run in manifests if run["context"]["resumed_from_run_id"])
    assert resumed["context"]["resumed_from_run_id"] == first["context"]["run_id"]
    assert resumed["result"]["epochs"][0]["epoch"] == 1
    assert resumed["config_snapshot"]["training"]["epochs"] == 2
