from pathlib import Path
from unittest.mock import MagicMock

import pytest
from bovi_core.config import Config
from testdata.mock_project import MOCK_PYPROJECT_TOML_ELABORATE

# Import moved inside functions to work with pytest pythonpath configuration


def get_project_structure(project_root: Path):
    """Helper function to define project directory structure (versioned)"""
    return {
        "src": project_root / "src",
        "bovi_core": project_root / "src" / "bovi_core",
        "experiments": project_root / "data" / "experiments" / "mock_exp",
        "config_dir": project_root
        / "data"
        / "experiments"
        / "mock_exp"
        / "versions"
        / "v1"
        / "config",
        "config_file": project_root
        / "data"
        / "experiments"
        / "mock_exp"
        / "versions"
        / "v1"
        / "config"
        / "config.yaml",
    }


# --- BASE PROJECT STRUCTURE FIXTURES ---


@pytest.fixture
def create_toml_at_root(tmp_path):
    """Creates pyproject.toml in a new project root directory"""
    project_root = tmp_path / "project_root"
    project_root.mkdir()

    toml_path = project_root / "pyproject.toml"
    toml_path.write_text(MOCK_PYPROJECT_TOML_ELABORATE)

    return project_root


@pytest.fixture
def create_config_in_experiments(create_toml_at_root):
    """Creates config.yaml in the versioned experiments directory structure"""
    project_root = create_toml_at_root
    structure = get_project_structure(project_root)

    # Create the src directory structure (ensure it exists for movement fixtures)
    structure["src"].mkdir(exist_ok=True)

    # Create the bovi_core directory structure
    structure["bovi_core"].mkdir(parents=True, exist_ok=True)

    # Create the versioned config directory structure
    structure["config_dir"].mkdir(parents=True, exist_ok=True)

    # Create config file
    from testdata.mock_config import MOCK_CONFIG_YAML_ELABORATE

    structure["config_file"].write_text(MOCK_CONFIG_YAML_ELABORATE)

    return project_root


@pytest.fixture
def mock_project_root(create_config_in_experiments, monkeypatch):
    """Creates complete mock project and changes to root directory"""
    project_root = create_config_in_experiments

    # Change to project root directory
    monkeypatch.chdir(project_root)

    return project_root


# --- DIRECTORY MOVEMENT FIXTURES ---


@pytest.fixture
def in_src_path(mock_project_root, monkeypatch):
    """Moves to src directory within the project"""
    src_path = mock_project_root / "src"
    monkeypatch.chdir(src_path)
    return src_path


@pytest.fixture
def in_sub_dir(mock_project_root, monkeypatch):
    """Moves to a subdirectory within src"""
    sub_path = mock_project_root / "src" / "subdir"
    # Ensure parent directories exist
    sub_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(sub_path)
    return sub_path


@pytest.fixture
def in_ssub_dir(mock_project_root, monkeypatch):
    """Moves to a sub-subdirectory within src"""
    ssub_path = mock_project_root / "src" / "subdir" / "subsubdir"
    # Ensure parent directories exist
    ssub_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(ssub_path)
    return ssub_path


@pytest.fixture
def in_parent_dir(mock_project_root, monkeypatch):
    """Moves to parent directory of the project root"""
    parent_path = mock_project_root.parent
    monkeypatch.chdir(parent_path)
    return parent_path


@pytest.fixture
def in_pparent_dir(mock_project_root, monkeypatch):
    """Moves to parent-parent directory of the project root"""
    pparent_path = mock_project_root.parent.parent
    monkeypatch.chdir(pparent_path)
    return pparent_path


# --- CONFIG FIXTURES ---
@pytest.fixture
def config_setup(mock_project_root, monkeypatch):
    """A fixture to create a temporary directory with elaborate mock config files."""
    project_root = mock_project_root
    structure = get_project_structure(project_root)

    # Create bovi_core directory structure using helper function
    structure["bovi_core"].mkdir(parents=True, exist_ok=True)

    # Add project's src to Python path using monkeypatch
    monkeypatch.syspath_prepend(str(project_root / "src"))

    # We must reset the singleton instance for every test to ensure isolation
    Config._instance = None

    # Initialize config with the paths to the temporary files
    config = Config(
        config_file_path=str(structure["config_file"]),
        project_file_path=str(project_root / "pyproject.toml"),
    )
    return config


@pytest.fixture
def mock_dbutils():
    """Provides a MagicMock replacement for the dbutils object."""
    return MagicMock()
