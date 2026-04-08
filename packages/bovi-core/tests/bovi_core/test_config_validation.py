"""
Tests for Config validation with various error scenarios.

Tests cover:
- Invalid experiment folder names
- Experiment name mismatches between folder and config
- Missing/invalid config files
- Missing required fields in config
- Empty/invalid YAML files
- Singleton behavior
"""

import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from bovi_core.config import Config


class TestConfigExperimentNotFound:
    """Tests for missing experiment directories/files"""

    def test_config_with_invalid_experiment_name(self):
        """Test that invalid experiment name raises FileNotFoundError with helpful message"""
        with pytest.raises(FileNotFoundError) as exc_info:
            Config("yolo_experiment")  # Doesn't exist

        error_msg = str(exc_info.value)
        # Should mention the experiment name
        assert "yolo_experiment" in error_msg
        # Should mention the expected directory structure
        assert "data/experiments/" in error_msg
        # Should list available experiments
        assert "Available experiments:" in error_msg

    def test_config_with_nonexistent_config_file(self):
        """Test that non-existent config_file_name raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError) as exc_info:
            # Folder exists (first_yolo_experiment), but dev_config.yaml doesn't
            Config("first_yolo_experiment", config_file_name="dev_config.yaml")

        error_msg = str(exc_info.value)
        assert "dev_config.yaml" in error_msg or "dev_config" in error_msg
        assert "first_yolo_experiment" in error_msg
        # Should list available config files in the experiment folder
        assert "Available config files" in error_msg or "config.yaml" in error_msg

    def test_config_with_multiple_configs_validates_each(self, tmp_path):
        """Test that validation works correctly when multiple config files exist"""
        # Setup: Create experiment with multiple config files (versioned structure)
        project_root = tmp_path / "project"
        project_root.mkdir()

        pyproject_path = project_root / "pyproject.toml"
        pyproject_path.write_text(
            "[project]\n"
            "name = \"test_project\"\n"
            "workspace_user = \"shared\"\n"
            "authors = [{ name = 'Test User', email = 'test@example.com' }]\n"
        )

        # Create versioned config directory
        config_dir = project_root / "data" / "experiments" / "multi_config_exp" / "versions" / "v1" / "config"
        config_dir.mkdir(parents=True)

        # Create config.yaml with correct experiment_name
        config1 = config_dir / "config.yaml"
        config1.write_text(yaml.dump({
            "experiment_name": "multi_config_exp",  # Matches folder
            "experiment_version": 1
        }))

        # Create dev.yaml with WRONG experiment_name
        config2 = config_dir / "dev.yaml"
        config2.write_text(yaml.dump({
            "experiment_name": "wrong_name",  # Does NOT match folder
            "experiment_version": 2
        }))

        # Reset singleton before each test
        Config._instance = None
        Config._instance_params = None

        # Test 1: Loading config.yaml should succeed (correct name)
        config_good = Config(
            "multi_config_exp",
            config_file_name="config.yaml",
            project_file_path=str(pyproject_path),
        )
        assert config_good.experiment.experiment_name == "multi_config_exp"
        assert config_good.experiment.experiment_version == 1

        # Reset singleton
        Config._instance = None
        Config._instance_params = None

        # Test 2: Loading dev.yaml should fail (wrong name)
        with pytest.raises(ValueError) as exc_info:
            Config(
                "multi_config_exp",
                config_file_name="dev.yaml",
                project_file_path=str(pyproject_path),
            )

        error_msg = str(exc_info.value)
        assert "mismatch" in error_msg.lower()
        assert "multi_config_exp" in error_msg  # Folder name
        assert "wrong_name" in error_msg  # Config experiment_name
        assert "Folder name" in error_msg


class TestConfigValidationErrors:
    """Tests for validation errors in config content"""

    def test_config_validates_experiment_name_match(self, tmp_path):
        """Test that config file experiment_name must match folder name"""
        # Setup: Create test config with mismatched experiment_name (versioned structure)
        project_root = tmp_path / "project"
        project_root.mkdir()

        pyproject_path = project_root / "pyproject.toml"
        pyproject_path.write_text(
            "[project]\n"
            "name = \"test_project\"\n"
            "workspace_user = \"shared\"\n"
            "authors = [{ name = 'Test User', email = 'test@example.com' }]\n"
        )

        # Create versioned config directory with mismatched name
        config_dir = project_root / "data" / "experiments" / "test_folder_name" / "versions" / "v1" / "config"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        # Config has different experiment_name than folder
        config_file.write_text(yaml.dump({
            "experiment_name": "different_config_name",
            "experiment_version": 1
        }))

        with pytest.raises(ValueError) as exc_info:
            Config(
                "test_folder_name",
                project_file_path=str(pyproject_path),
            )

        error_msg = str(exc_info.value)
        # Should mention the mismatch
        assert "mismatch" in error_msg.lower()
        # Should show folder name
        assert "test_folder_name" in error_msg
        # Should show config experiment_name
        assert "different_config_name" in error_msg
        # Should mention "Folder name"
        assert "Folder name" in error_msg or "folder" in error_msg.lower()
        # Should provide helpful suggestions
        assert "Rename" in error_msg or "Update" in error_msg

    def test_config_with_empty_yaml(self, tmp_path):
        """Test that empty YAML file raises ValueError"""
        # Setup: Create minimal pyproject.toml
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Write minimal pyproject.toml
        pyproject_path = project_root / "pyproject.toml"
        pyproject_path.write_text(
            "[project]\n"
            "name = \"test_project\"\n"
            "workspace_user = \"shared\"\n"
            "authors = [{ name = 'Test User', email = 'test@example.com' }]\n"
        )

        # Create versioned config directory with empty config
        config_dir = project_root / "data" / "experiments" / "test_empty" / "versions" / "v1" / "config"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        config_file.write_text("")  # Empty file

        with pytest.raises(ValueError) as exc_info:
            Config(
                "test_empty",
                project_file_path=str(pyproject_path),
            )

        assert "empty" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

    def test_config_with_missing_experiment_name(self, tmp_path):
        """Test that missing experiment_name raises ValueError"""
        # Setup
        project_root = tmp_path / "project"
        project_root.mkdir()

        pyproject_path = project_root / "pyproject.toml"
        pyproject_path.write_text(
            "[project]\n"
            "name = \"test_project\"\n"
            "workspace_user = \"shared\"\n"
            "authors = [{ name = 'Test User', email = 'test@example.com' }]\n"
        )

        # Create versioned config without experiment_name
        config_dir = project_root / "data" / "experiments" / "test_missing_name" / "versions" / "v1" / "config"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        config_file.write_text(yaml.dump({"experiment_version": 1}))

        with pytest.raises(ValueError) as exc_info:
            Config(
                "test_missing_name",
                project_file_path=str(pyproject_path),
            )

        error_msg = str(exc_info.value)
        assert "experiment_name" in error_msg
        assert "missing" in error_msg.lower() or "required" in error_msg.lower()

    def test_config_with_missing_experiment_version(self, tmp_path):
        """Test that missing experiment_version raises ValueError"""
        # Setup
        project_root = tmp_path / "project"
        project_root.mkdir()

        pyproject_path = project_root / "pyproject.toml"
        pyproject_path.write_text(
            "[project]\n"
            "name = \"test_project\"\n"
            "workspace_user = \"shared\"\n"
            "authors = [{ name = 'Test User', email = 'test@example.com' }]\n"
        )

        # Create versioned config without experiment_version
        config_dir = project_root / "data" / "experiments" / "test_missing_ver" / "versions" / "v1" / "config"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        config_file.write_text(yaml.dump({"experiment_name": "test_missing_ver"}))

        with pytest.raises(ValueError) as exc_info:
            Config(
                "test_missing_ver",
                project_file_path=str(pyproject_path),
            )

        error_msg = str(exc_info.value)
        assert "experiment_version" in error_msg
        assert "missing" in error_msg.lower() or "required" in error_msg.lower()


class TestConfigSingleton:
    """Tests for Config singleton behavior"""

    def test_config_singleton_returns_same_instance(self, tmp_path):
        """Test that same params return same instance"""
        # Reset singleton
        Config._instance = None
        Config._instance_params = None

        # Setup: Create test config with matching experiment_name (versioned structure)
        project_root = tmp_path / "project"
        project_root.mkdir()

        pyproject_path = project_root / "pyproject.toml"
        pyproject_path.write_text(
            "[project]\n"
            "name = \"test_project\"\n"
            "workspace_user = \"shared\"\n"
            "authors = [{ name = 'Test User', email = 'test@example.com' }]\n"
        )

        # Create versioned config directory
        config_dir = project_root / "data" / "experiments" / "test_singleton" / "versions" / "v1" / "config"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        config_file.write_text(yaml.dump({
            "experiment_name": "test_singleton",
            "experiment_version": 1
        }))

        # First config
        config1 = Config(
            "test_singleton",
            project_file_path=str(pyproject_path),
        )

        # Second call with same params should return same instance (no reset)
        config2 = Config(
            "test_singleton",
            project_file_path=str(pyproject_path),
        )

        assert config1 is config2, "Singleton should return same instance for same params"

    def test_config_requires_experiment_or_path(self):
        """Test that Config() without params uses existing singleton or raises error"""
        # Reset singleton
        Config._instance = None
        Config._instance_params = None

        # First call without params should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            Config()

        assert "experiment_name or config_file_path must be provided" in str(exc_info.value)


class TestConfigValidationMessages:
    """Tests to verify error messages are helpful"""

    def test_error_message_shows_expected_path(self):
        """Test that error messages clearly show the expected path"""
        with pytest.raises(FileNotFoundError) as exc_info:
            Config("nonexistent_xyz_123")

        error_msg = str(exc_info.value)
        # Should show the full expected path
        assert "Expected path:" in error_msg
        assert "nonexistent_xyz_123" in error_msg

    def test_error_message_shows_available_experiments(self, tmp_path):
        """Test that error messages list available experiments."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        pyproject = project_root / "pyproject.toml"
        pyproject.write_text(
            "[project]\nname = \"test_project\"\n"
            "authors = [{ name = 'Test User', email = 'test@example.com' }]\n"
        )
        # Create an existing experiment so it shows up in "Available experiments"
        exp_dir = project_root / "data" / "experiments" / "first_yolo_experiment" / "versions" / "v1" / "config"
        exp_dir.mkdir(parents=True)
        (exp_dir / "config.yaml").write_text("experiment_name: first_yolo_experiment\nexperiment_version: 1\n")

        Config._instance = None
        with pytest.raises(FileNotFoundError) as exc_info:
            Config("does_not_exist_xyz", project_file_path=str(pyproject))

        error_msg = str(exc_info.value)
        assert "Available experiments:" in error_msg
        assert "first_yolo_experiment" in error_msg

    def test_mismatch_error_explains_both_names(self, tmp_path):
        """Test that mismatch error clearly shows folder vs config names"""
        # Setup: Create test config with mismatched experiment_name (versioned structure)
        project_root = tmp_path / "project"
        project_root.mkdir()

        pyproject_path = project_root / "pyproject.toml"
        pyproject_path.write_text(
            "[project]\n"
            "name = \"test_project\"\n"
            "workspace_user = \"shared\"\n"
            "authors = [{ name = 'Test User', email = 'test@example.com' }]\n"
        )

        config_dir = project_root / "data" / "experiments" / "actual_folder" / "versions" / "v1" / "config"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        config_file.write_text(yaml.dump({
            "experiment_name": "wrong_name_in_config",
            "experiment_version": 1
        }))

        with pytest.raises(ValueError) as exc_info:
            Config(
                "actual_folder",
                project_file_path=str(pyproject_path),
            )

        error_msg = str(exc_info.value)
        # Should clearly show folder name
        assert "actual_folder" in error_msg
        # Should clearly show config experiment_name
        assert "wrong_name_in_config" in error_msg
        # Should mention it's a mismatch
        assert "mismatch" in error_msg.lower()
        # Should distinguish between folder and config
        assert "Folder name" in error_msg or "folder" in error_msg.lower()
        assert "config" in error_msg.lower() or "experiment_name" in error_msg

    def test_error_lists_available_config_files(self, tmp_path):
        """Test that error messages list available config files in experiment folder."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        pyproject = project_root / "pyproject.toml"
        pyproject.write_text(
            "[project]\nname = \"test_project\"\n"
            "authors = [{ name = 'Test User', email = 'test@example.com' }]\n"
        )
        # Create experiment with config.yaml
        config_dir = project_root / "data" / "experiments" / "first_yolo_experiment" / "versions" / "v1" / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "config.yaml").write_text("experiment_name: first_yolo_experiment\nexperiment_version: 1\n")

        Config._instance = None
        with pytest.raises(FileNotFoundError) as exc_info:
            Config("first_yolo_experiment", config_file_name="production.yaml", project_file_path=str(pyproject))

        error_msg = str(exc_info.value)
        assert "production.yaml" in error_msg
        assert "Available config files" in error_msg
        assert "config.yaml" in error_msg
