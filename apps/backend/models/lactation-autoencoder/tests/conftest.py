"""Shared test fixtures."""

from pathlib import Path

import pytest
from bovi_core.utils.path_utils import get_project_root


@pytest.fixture
def project_root():
    return Path(get_project_root(project_name="bovi"))
