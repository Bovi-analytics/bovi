"""Tests for path_utils module."""

from pathlib import Path

import pytest
from bovi_core.utils.path_utils import (
    _extract_version_number,
    get_experiment_paths,
    get_project_root,
    get_run_config_path,
    make_path_absolute,
)


class TestExtractVersionNumber:
    """Tests for _extract_version_number function."""

    def test_single_digit_version(self):
        """Test extraction of single digit versions."""
        assert _extract_version_number("v1") == 1
        assert _extract_version_number("v2") == 2
        assert _extract_version_number("v9") == 9

    def test_multi_digit_version(self):
        """Test extraction of multi-digit versions."""
        assert _extract_version_number("v10") == 10
        assert _extract_version_number("v11") == 11
        assert _extract_version_number("v100") == 100
        assert _extract_version_number("v999") == 999

    def test_non_standard_format_returns_zero(self):
        """Test that non-standard formats return 0."""
        assert _extract_version_number("version1") == 0
        assert _extract_version_number("1") == 0
        assert _extract_version_number("") == 0
        assert _extract_version_number("latest") == 0

    def test_version_sorting(self):
        """Test that versions sort correctly by numeric value."""
        versions = ["v1", "v11", "v2", "v100", "v9"]
        sorted_versions = sorted(versions, key=_extract_version_number)
        assert sorted_versions == ["v1", "v2", "v9", "v11", "v100"]

        # max should return highest numeric version
        assert max(versions, key=_extract_version_number) == "v100"


class TestGetRunConfigPath:
    """Tests for get_run_config_path function."""

    def test_selects_version_with_config(self, mock_project_root):
        """Test that version with config is selected."""
        # mock_project_root has v1 with config.yaml
        result = get_run_config_path(str(mock_project_root), "mock_exp")

        assert "v1" in result
        assert result.endswith("config.yaml")

    def test_explicit_version_override(self, mock_project_root):
        """Test that explicit version parameter is respected."""
        result = get_run_config_path(str(mock_project_root), "mock_exp", version="v1")

        assert "v1" in result

    def test_numeric_sorting_prefers_higher_version_with_config(self, mock_project_root):
        """Test that v11 is selected over v2 when both have configs."""
        versions_dir = mock_project_root / "data" / "experiments" / "mock_exp" / "versions"

        # Create v2 with config
        v2_config = versions_dir / "v2" / "config"
        v2_config.mkdir(parents=True, exist_ok=True)
        (v2_config / "config.yaml").write_text("experiment_name: mock_exp\nexperiment_version: 2")

        # Create v11 with config
        v11_config = versions_dir / "v11" / "config"
        v11_config.mkdir(parents=True, exist_ok=True)
        (v11_config / "config.yaml").write_text("experiment_name: mock_exp\nexperiment_version: 11")

        result = get_run_config_path(str(mock_project_root), "mock_exp")

        # v11 should be selected (highest with config, numeric sorting)
        assert "v11" in result

    def test_skips_version_without_config(self, mock_project_root):
        """Test that versions without config.yaml are skipped."""
        versions_dir = mock_project_root / "data" / "experiments" / "mock_exp" / "versions"

        # Create v11 WITHOUT config (empty directory)
        v11_config = versions_dir / "v11" / "config"
        v11_config.mkdir(parents=True, exist_ok=True)
        # No config.yaml file

        result = get_run_config_path(str(mock_project_root), "mock_exp")

        # Should select v1 (has config), not v11 (no config)
        assert "v1" in result
        assert "v11" not in result

    def test_fallback_when_no_configs_exist(self, mock_project_root):
        """Test fallback behavior when no version has a config file."""
        # Create a new experiment with versions but no configs
        versions_dir = mock_project_root / "data" / "experiments" / "empty_exp" / "versions"
        (versions_dir / "v1" / "config").mkdir(parents=True)
        (versions_dir / "v5" / "config").mkdir(parents=True)

        result = get_run_config_path(str(mock_project_root), "empty_exp")

        # Should fallback to highest version numerically
        assert "v5" in result

    def test_custom_config_file_name(self, mock_project_root):
        """Test using a custom config file name."""
        versions_dir = mock_project_root / "data" / "experiments" / "mock_exp" / "versions"
        v1_config = versions_dir / "v1" / "config"

        # Create custom config file
        (v1_config / "custom.yaml").write_text("experiment_name: mock_exp\nexperiment_version: 1")

        result = get_run_config_path(
            str(mock_project_root), "mock_exp", config_file_name="custom.yaml"
        )

        assert "v1" in result
        assert result.endswith("custom.yaml")


class TestMakePathAbsolute:
    """Tests for make_path_absolute function."""

    def test_absolute_path_unchanged(self):
        """Test that absolute paths are returned as-is."""
        abs_path = "/absolute/path/to/file"
        result = make_path_absolute(abs_path)
        assert str(result) == abs_path

    def test_relative_path_resolved(self, mock_project_root):
        """Test that relative paths are resolved to project root."""
        result = make_path_absolute("data/file.txt", project_root=mock_project_root)

        assert result.is_absolute()
        assert str(result) == str(mock_project_root / "data" / "file.txt")

    def test_path_object_input(self, mock_project_root):
        """Test that Path objects work as input."""
        result = make_path_absolute(Path("data/file.txt"), project_root=mock_project_root)

        assert result.is_absolute()
        assert "data" in str(result)
        assert "file.txt" in str(result)


class TestGetExperimentPaths:
    """Tests for get_experiment_paths function."""

    def test_returns_all_expected_keys(self, mock_project_root):
        """Test that all expected paths are returned."""
        result = get_experiment_paths(str(mock_project_root), "test_exp", "v1")

        assert "experiments_dir" in result
        assert "dir" in result
        assert "config_dir" in result
        assert "weights_dir" in result

    def test_paths_are_absolute(self, mock_project_root):
        """Test that all returned paths are absolute."""
        result = get_experiment_paths(str(mock_project_root), "test_exp", "v1")

        for key, path in result.items():
            assert Path(path).is_absolute(), f"{key} should be absolute"

    def test_version_normalization_with_v_prefix(self, mock_project_root):
        """Test that version with 'v' prefix is handled correctly."""
        result = get_experiment_paths(str(mock_project_root), "test_exp", "v1")

        assert "v1" in result["dir"]

    def test_version_normalization_without_v_prefix(self, mock_project_root):
        """Test that version without 'v' prefix is normalized."""
        result = get_experiment_paths(str(mock_project_root), "test_exp", "1")

        # Should be normalized to v1
        assert "v1" in result["dir"]

    def test_version_normalization_with_decimal(self, mock_project_root):
        """Test that decimal version is handled."""
        result = get_experiment_paths(str(mock_project_root), "test_exp", "1.0")

        # Should be normalized to v1.0
        assert "v1.0" in result["dir"]

    def test_correct_directory_structure(self, mock_project_root):
        """Test that paths follow correct structure."""
        result = get_experiment_paths(str(mock_project_root), "my_experiment", "v2")

        assert result["experiments_dir"].endswith("data/experiments")
        assert "my_experiment/versions/v2" in result["dir"]
        assert result["config_dir"].endswith("config")
        assert result["weights_dir"].endswith("weights")


def _write_toml(path: Path, project_name: str) -> None:
    """Helper: write a minimal pyproject.toml with the given project name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'[project]\nname = "{project_name}"\n')


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_no_project_name_finds_first_toml(self, tmp_path, monkeypatch):
        """Without project_name, returns the first pyproject.toml found (cwd)."""
        _write_toml(tmp_path / "pyproject.toml", "some-project")
        monkeypatch.chdir(tmp_path)

        assert get_project_root() == str(tmp_path)

    def test_project_name_matches_cwd_toml(self, tmp_path, monkeypatch):
        """When cwd's toml matches project_name, returns cwd."""
        _write_toml(tmp_path / "pyproject.toml", "my-project")
        monkeypatch.chdir(tmp_path)

        assert get_project_root(project_name="my-project") == str(tmp_path)

    def test_project_name_skips_non_matching_cwd_toml(self, tmp_path, monkeypatch):
        """When cwd's toml does NOT match, keeps searching."""
        # Workspace root toml (wrong name)
        _write_toml(tmp_path / "pyproject.toml", "workspace-root")
        # Package toml (correct name) in a subdirectory
        pkg_dir = tmp_path / "packages" / "my-pkg"
        _write_toml(pkg_dir / "pyproject.toml", "my-pkg")

        monkeypatch.chdir(tmp_path)

        assert get_project_root(project_name="my-pkg") == str(pkg_dir)

    def test_walks_up_to_find_matching_parent_toml(self, tmp_path, monkeypatch):
        """Walks up from a subdirectory to find the matching toml."""
        _write_toml(tmp_path / "pyproject.toml", "my-project")
        sub_dir = tmp_path / "src" / "deep" / "nested"
        sub_dir.mkdir(parents=True)
        monkeypatch.chdir(sub_dir)

        assert get_project_root(project_name="my-project") == str(tmp_path)

    def test_walks_up_skips_wrong_parent_toml(self, tmp_path, monkeypatch):
        """Walks up past a non-matching parent toml to find the correct one."""
        # Monorepo root
        _write_toml(tmp_path / "pyproject.toml", "monorepo")
        # Package dir with its own toml
        pkg_dir = tmp_path / "packages" / "my-pkg"
        _write_toml(pkg_dir / "pyproject.toml", "my-pkg")
        # cwd is inside the package's src
        src_dir = pkg_dir / "src" / "my_pkg"
        src_dir.mkdir(parents=True)
        monkeypatch.chdir(src_dir)

        # Should find my-pkg's toml, not the monorepo root's toml
        assert get_project_root(project_name="my-pkg") == str(pkg_dir)

    def test_monorepo_from_root_finds_correct_package(self, tmp_path, monkeypatch):
        """From monorepo root, walks down to find the right package toml."""
        _write_toml(tmp_path / "pyproject.toml", "bovi")
        pkg_yolo = tmp_path / "packages" / "models" / "bovi-yolo"
        pkg_ae = tmp_path / "packages" / "models" / "lactation-autoencoder"
        _write_toml(pkg_yolo / "pyproject.toml", "bovi-yolo")
        _write_toml(pkg_ae / "pyproject.toml", "lactation-autoencoder")

        monkeypatch.chdir(tmp_path)

        assert get_project_root(project_name="bovi-yolo") == str(pkg_yolo)
        assert get_project_root(project_name="lactation-autoencoder") == str(pkg_ae)

    def test_monorepo_from_package_src_finds_package_not_root(self, tmp_path, monkeypatch):
        """From inside a package's src dir, finds that package — not the workspace root."""
        _write_toml(tmp_path / "pyproject.toml", "bovi")
        pkg_dir = tmp_path / "packages" / "models" / "bovi-yolo"
        _write_toml(pkg_dir / "pyproject.toml", "bovi-yolo")
        src_dir = pkg_dir / "src" / "bovi_yolo"
        src_dir.mkdir(parents=True)
        monkeypatch.chdir(src_dir)

        assert get_project_root(project_name="bovi-yolo") == str(pkg_dir)

    def test_project_name_not_found_raises(self, tmp_path, monkeypatch):
        """Raises ValueError when no pyproject.toml matches the project_name."""
        _write_toml(tmp_path / "pyproject.toml", "other-project")
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="nonexistent-project"):
            get_project_root(project_name="nonexistent-project")

    def test_no_toml_at_all_raises(self, tmp_path, monkeypatch):
        """Raises ValueError when there is no pyproject.toml anywhere."""
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="Project root not found"):
            get_project_root()

    def test_no_project_name_returns_nearest_toml_walking_up(self, tmp_path, monkeypatch):
        """Without project_name, returns the nearest parent with a toml (backward compat)."""
        _write_toml(tmp_path / "pyproject.toml", "root-project")
        sub_dir = tmp_path / "a" / "b" / "c"
        sub_dir.mkdir(parents=True)
        monkeypatch.chdir(sub_dir)

        # Should find root-project's toml since it's the first one walking up
        assert get_project_root() == str(tmp_path)
