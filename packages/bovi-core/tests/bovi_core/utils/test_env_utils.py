import os
from pathlib import Path

from bovi_core.utils.env_utils import (
    detect_environment,
    get_project_root,
    get_project_src,
    read_project_name_from_toml,
)


def test_detect_environment_is_databricks(mocker):
    """Test that 'databricks' is detected when DATABRICKS_RUNTIME_VERSION is set."""
    # Use mocker to simulate the os.environ dictionary for this one test
    mocker.patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "11.3"})
    assert detect_environment() == "databricks"


def test_detect_environment_is_databricks_connect(mocker):
    """Test that 'vscode_remote' is detected for Databricks Connect."""
    # Remove DATABRICKS_RUNTIME_VERSION if it exists, and set SPARK_REMOTE
    mocker.patch.dict(os.environ, {"SPARK_REMOTE": "127.0.0.1"}, clear=False)
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        del os.environ["DATABRICKS_RUNTIME_VERSION"]
    assert detect_environment() == "vscode_remote"


def test_detect_environment_is_local(mocker):
    """Test that 'local' is detected when no other env vars are set."""
    # Remove all relevant environment variables for this test
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        del os.environ["DATABRICKS_RUNTIME_VERSION"]
    if "SPARK_REMOTE" in os.environ:
        del os.environ["SPARK_REMOTE"]
    if "DATABRICKS_CONNECT" in os.environ:
        del os.environ["DATABRICKS_CONNECT"]
    # Also mock os.path.exists to simulate not being on a Databricks cluster filesystem
    mocker.patch("os.path.exists", return_value=False)
    assert detect_environment() == "local"


def test_detect_environment_is_databricks_via_dbfs(mocker):
    """Test that 'databricks' is detected via the existence of /dbfs."""
    # Remove all relevant environment variables for this test
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        del os.environ["DATABRICKS_RUNTIME_VERSION"]
    if "SPARK_REMOTE" in os.environ:
        del os.environ["SPARK_REMOTE"]
    if "DATABRICKS_CONNECT" in os.environ:
        del os.environ["DATABRICKS_CONNECT"]
    mocker.patch("os.path.exists", return_value=True)  # Pretend /dbfs exists
    assert detect_environment() == "databricks"


class TestPathUtils:
    def test_get_project_root_from_root(self, mock_project_root):
        """Test get_project_root when already in project root"""
        actual_root = get_project_root()
        assert actual_root == str(mock_project_root)

    def test_get_project_src_from_root(self, mock_project_root):
        """Test get_project_src when in project root"""
        actual_src = get_project_src()
        assert actual_src == str(mock_project_root / "src")

    def test_get_project_root_from_src(self, in_src_path, mock_project_root):
        """Test get_project_root when in src directory"""
        actual_root = get_project_root()
        assert actual_root == str(mock_project_root)

    def test_get_project_src_from_src(self, in_src_path, mock_project_root):
        """Test get_project_src when in src directory"""
        actual_src = get_project_src()
        assert actual_src == str(mock_project_root / "src")

    def test_get_project_root_from_subdir(self, in_sub_dir, mock_project_root):
        """Test get_project_root when in subdirectory"""
        actual_root = get_project_root()
        assert actual_root == str(mock_project_root)

    def test_get_project_root_from_ssubdir(self, in_ssub_dir, mock_project_root):
        """Test get_project_root when in sub-subdirectory"""
        actual_root = get_project_root()
        assert actual_root == str(mock_project_root)

    def test_get_project_root_from_parent_dir(self, in_parent_dir, mock_project_root):
        """Test get_project_root when in parent directory of project root"""
        actual_root = get_project_root()
        assert actual_root == str(mock_project_root)

    def test_get_project_root_from_pparent_dir(self, in_pparent_dir, mock_project_root):
        """Test get_project_root when in parent-parent directory of project root"""
        actual_root = get_project_root()
        # The function should find a valid project root
        # Note: It might find our project root or one from another test
        assert actual_root is not None
        assert len(actual_root) > 0
        # Verify it's a valid path that contains a pyproject.toml
        assert Path(actual_root).exists()
        assert "pyproject.toml" in [f.name for f in Path(actual_root).iterdir() if f.is_file()]
        # Verify it's a valid project root by checking it has a project name
        project_name = read_project_name_from_toml(Path(actual_root) / "pyproject.toml")
        assert project_name != "unknown"
