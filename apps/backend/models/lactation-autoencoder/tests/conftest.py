"""Shared test fixtures."""

from pathlib import Path

import pytest
from bovi_core.utils.path_utils import get_project_root

_WEIGHTS = (
    Path(get_project_root(project_name="bovi"))
    / "data/models/lactation_autoencoder/versions/v15/weights/autoencoder/saved_model.pb"
)

# Skip the entire suite when the TF SavedModel isn't available locally —
# weights live in Azure Blob Storage, not git, so CI can't load them.
collect_ignore_glob = [] if _WEIGHTS.exists() else ["test_main.py"]


@pytest.fixture
def project_root():
    return Path(get_project_root(project_name="bovi"))
