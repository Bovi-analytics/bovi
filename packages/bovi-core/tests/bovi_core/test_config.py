from pathlib import Path

import pytest

# We need to ensure we can import the class we're testing
from bovi_core.config import Config
from testdata.mock_config import MOCK_CONFIG_YAML_ELABORATE
from testdata.mock_project import MOCK_PYPROJECT_TOML_ELABORATE


# --- TEST CLASSES ---
class TestConfigInitialization:
    """Tests focused on the creation and loading process of the Config object."""

    def test_successful_initialization(self, config_setup):
        """Happy path: ensure both project and run configs are loaded."""
        config = config_setup
        assert config is not None
        assert config.project is not None
        assert config.experiment is not None
        assert config.environment == "local"  # Pytest runs in a local env

    def test_file_not_found_errors(self, tmp_path):
        """Test that the system raises appropriate errors for missing files and invalid parameters."""
        Config._instance = None

        # Test missing run file (auto-resolved from experiment name, but experiment doesn't exist)
        with pytest.raises(FileNotFoundError, match="Run config file not found"):
            Config(experiment_name="non_existent_experiment")

        # Test missing run file (with a valid project file but non-existent experiment)
        Config._instance = None
        project_file_path = tmp_path / "pyproject.toml"
        project_file_path.write_text(MOCK_PYPROJECT_TOML_ELABORATE)
        with pytest.raises(FileNotFoundError, match="Run config file not found"):
            Config(experiment_name="non_existent_experiment", project_file_path=project_file_path)

        # Test missing run file with explicit path
        Config._instance = None
        with pytest.raises(FileNotFoundError, match="Run config file not found"):
            Config(
                config_file_path=tmp_path / "non_existent.yaml", project_file_path=project_file_path
            )

    def test_missing_project_file_error(self, tmp_path):
        """Test that the system raises ValueError for missing project file."""
        Config._instance = None

        # Change to a temp directory where no pyproject.toml exists
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with pytest.raises(ValueError, match="Project root not found"):
                Config(experiment_name="test_exp")
        finally:
            os.chdir(original_cwd)

    def test_parameter_validation_errors(self, tmp_path):
        """Test that the system raises ValueError for invalid parameter combinations."""
        Config._instance = None

        # Test neither experiment_name nor config_file_path provided
        project_file_path = tmp_path / "pyproject.toml"
        project_file_path.write_text(MOCK_PYPROJECT_TOML_ELABORATE)
        with pytest.raises(ValueError, match="experiment_name or config_file_path must be provided"):
            Config(project_file_path=project_file_path)

    def test_successful_initialization_with_experiment_name(self, tmp_path):
        """Test successful initialization using only experiment_name."""
        Config._instance = None

        # Create project structure
        project_file_path = tmp_path / "pyproject.toml"
        project_file_path.write_text(MOCK_PYPROJECT_TOML_ELABORATE)

        # Create versioned experiment directory and config with matching experiment_name
        config_dir = tmp_path / "data" / "experiments" / "test_exp" / "versions" / "v1" / "config"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        # Use inline config with matching experiment_name
        config_file.write_text("experiment_name: test_exp\nexperiment_version: 1\nbatch_size: 32")

        # Should successfully initialize
        config = Config(experiment_name="test_exp", project_file_path=project_file_path)
        assert config is not None
        assert config.experiment_name == "test_exp"
        assert config.experiment is not None

    def test_successful_initialization_with_config_file_path_only(self, tmp_path):
        """Test successful initialization using only config_file_path (experiment_name extracted)."""
        Config._instance = None

        # Create project structure
        project_file_path = tmp_path / "pyproject.toml"
        project_file_path.write_text(MOCK_PYPROJECT_TOML_ELABORATE)

        # Create experiment directory and config with matching experiment_name
        exp_dir = tmp_path / "extracted_exp"
        exp_dir.mkdir()
        config_file = exp_dir / "config.yaml"
        # Use inline config with matching experiment_name
        config_file.write_text("experiment_name: extracted_exp\nexperiment_version: 1\nbatch_size: 32")

        # Should successfully initialize and extract experiment_name from path
        config = Config(config_file_path=config_file, project_file_path=project_file_path)
        assert config is not None
        assert config.experiment_name == "extracted_exp"  # Should be extracted from path
        assert config.experiment is not None

    def test_malformed_file_errors(self, tmp_path):
        """Test that the system raises RuntimeError for malformed files."""
        Config._instance = None
        project_file_path = tmp_path / "pyproject.toml"
        config_file_path = tmp_path / "config.yaml"

        # Test malformed TOML
        project_file_path.write_text("this is not valid toml")
        config_file_path.write_text(MOCK_CONFIG_YAML_ELABORATE)
        with pytest.raises(RuntimeError, match="Error loading project file"):
            Config(config_file_path=config_file_path, project_file_path=project_file_path)

        # Test malformed YAML
        Config._instance = None
        project_file_path.write_text(MOCK_PYPROJECT_TOML_ELABORATE)
        config_file_path.write_text("models: yolo: - invalid yaml")
        with pytest.raises(RuntimeError, match="Error loading run config file"):
            Config(config_file_path=config_file_path, project_file_path=project_file_path)


class TestAttributeAccessAndImmutability:
    """Tests focused on getting data and enforcing the read-only/write rules."""

    def test_nested_attribute_access(self, config_setup, monkeypatch):
        """Can we access deeply nested attributes from the TOML?"""
        config = config_setup
        assert config.project.blob_storage.container_name == "testcontainer"

        # Secrets are resolved via SecretsManager. The TOML value "actual-secret-name-in-vault"
        # is the secret key that gets looked up. We need to set the environment variable
        # that the SecretsManager will look for.
        monkeypatch.setenv("ACTUAL_SECRET_NAME_IN_VAULT", "my-secret-value")
        assert config.project.secrets.my_secret_key == "my-secret-value"

    def test_project_paths_are_set(self, config_setup):
        """Test that all project path attributes are correctly set."""
        config = config_setup

        # Test that all path attributes exist and are strings
        assert hasattr(config.project, 'src_dir')
        assert hasattr(config.project, 'notebooks_dir')
        assert hasattr(config.project, 'data_dir')
        assert hasattr(config.project, 'project_root')

        # Test that paths are non-empty strings
        assert isinstance(config.project.src_dir, str) and config.project.src_dir
        assert isinstance(config.project.notebooks_dir, str) and config.project.notebooks_dir
        assert isinstance(config.project.data_dir, str) and config.project.data_dir
        assert isinstance(config.project.project_root, str) and config.project.project_root

        # Test that data_dir ends with 'data' (following same pattern as notebooks_dir)
        assert config.project.data_dir.endswith('data')
        assert config.project.notebooks_dir.endswith('notebooks')

    def test_project_config_is_immutable(self, config_setup):
        """CRITICAL: Test that we cannot overwrite project-level configuration."""
        config = config_setup

        with pytest.raises(ValueError, match="Cannot update 'name'"):
            config.project.name = "a_new_name"

        with pytest.raises(ValueError, match="Cannot update 'cluster_name'"):
            config.project.databricks.cluster_name = "a_new_cluster"

    def test_run_config_is_mutable(self, config_setup):
        """CRITICAL: Test that we CAN overwrite run-level configuration."""
        config = config_setup

        # This should work without raising an error
        try:
            config.experiment.batch_size = 128
            config.experiment.models.yolo.new_attribute = "it works"
        except ValueError:
            pytest.fail("Should not have raised ValueError when modifying run config.")

        assert config.experiment.batch_size == 128
        assert config.experiment.models.yolo.new_attribute == "it works"


class TestTemplatingLogic:
    """A deep dive into the path templating engine and its edge cases."""

    def test_basic_path_resolution(self, config_setup):
        """Test a standard, successful path resolution."""
        path = config_setup.experiment.models.yolo.weights_blob.best
        expected = "elaborate_test_project/models/yolo/weights/yolo_best.pt"
        if path != expected:
            print(f"path: {path}")
            print(f"expected: {expected}")
        assert path == expected

    def test_model_variable_override(self, config_setup):
        """Test that a model's 'vars' section overrides the defaults."""
        path = config_setup.experiment.models.snn.config_path.default
        # Note: model_name should be "snn_override", not "snn"
        expected = "elaborate_test_project/models/snn_override/config/snn_config.yml"
        if path != expected:
            print(f"path: {path}")
            print(f"expected: {expected}")
        assert path == expected

    def test_multiple_templates_for_one_source(self, config_setup):
        """Test that multiple templates ('weights_blob', 'temp_weights') are generated from one source."""
        yolo_model = config_setup.experiment.models.yolo

        # Check first template
        path1 = yolo_model.weights_blob.large
        expected1 = "elaborate_test_project/models/yolo/weights/yolo_large.pt"
        if path1 != expected1:
            print(f"path1: {path1}")
            print(f"expected1: {expected1}")
        assert path1 == expected1

        # Check second template
        path2 = yolo_model.temp_weights.large
        expected2 = "/local_disk0/tmp/yolo/weights/yolo_large.pt"
        if path2 != expected2:
            print(f"path2: {path2}")
            print(f"expected2: {expected2}")
        assert path2 == expected2

    def test_unresolved_template_variable_is_handled_safely(self, config_setup):
        """Test that safe_substitute leaves missing template variables untouched."""
        path = config_setup.experiment.models.yolo.unresolved_template.best
        # {unresolved_var} was not provided, so it should remain in the string
        expected = "/data/{unresolved_var}/yolo.dat"
        if path != expected:
            print(f"path: {path}")
            print(f"expected: {expected}")
        assert path == expected

    def test_model_with_no_template_vars(self, config_setup):
        """A model without `template_vars` should be loaded but have no generated paths."""
        simple_model = config_setup.experiment.models.simple_model

        assert simple_model.some_value == 123
        # It should NOT have attributes that would have been generated by templates
        assert not hasattr(simple_model, "weights_blob")
        assert not hasattr(simple_model, "config_path")

    def test_config_without_path_templates_section(self, tmp_path):
        """If the yaml has no `path_templates`, templating should be skipped entirely."""
        Config._instance = None

        # Create a new YAML that is valid but lacks the templates section
        yaml_without_templates = """
        experiment_name: no_templates_test
        experiment_version: 1
        models:
          yolo:
            template_vars: # This will be ignored
              weights_file:
                best: "yolo_best.pt"
        """

        project_file_path = tmp_path / "pyproject.toml"
        project_file_path.write_text(MOCK_PYPROJECT_TOML_ELABORATE)

        # Create experiment folder with matching name
        exp_dir = tmp_path / "no_templates_test"
        exp_dir.mkdir()
        config_file_path = exp_dir / "config.yaml"
        config_file_path.write_text(yaml_without_templates)

        config = Config(config_file_path=config_file_path, project_file_path=project_file_path)

        # The 'yolo' model should exist, but no templating should have occurred.
        # The `template_vars` dictionary should have been removed.
        assert hasattr(config.experiment.models, "yolo")
        assert not hasattr(config.experiment.models.yolo, "weights_blob")
        assert not hasattr(config.experiment.models.yolo, "template_vars")

    def test_template_with_mixed_resolved_and_unresolved_vars(self, config_setup):
        """
        Tests a template where some variables exist and others don't,
        to stress the custom str.format() fallback logic.
        """
        # Add a new template to the stored path_templates for this test
        path_templates = config_setup._path_templates.copy()
        path_templates["mixed_template"] = {
            "template": "{project_name}/data/{model_name}/{missing_var}",
            "uses": "weights_file",
        }

        # Re-process the models with the new template
        # In a real scenario you wouldn't do this, this is for testing the processing logic
        project_vars = config_setup._flatten_config_node(config_setup.project)

        # Create a simple test model with template_vars to test the mixed resolution
        test_models = {"yolo": {"template_vars": {"weights_file": {"best": "yolo_best.pt"}}}}

        processed_models = config_setup._process_templated_models(
            test_models, path_templates, project_vars
        )

        path = processed_models["yolo"]["mixed_template"]["best"]

        # The existing variables should be filled, the missing one should remain
        expected = "elaborate_test_project/data/yolo/{missing_var}"
        assert path == expected


# --- TEST SINGLETON PATTERN ---


class TestConfigSingleton:
    """Tests the singleton pattern of the Config class."""

    def test_same_params_return_same_instance(self, config_setup):
        """
        Calling Config() with the same parameters twice should return the exact same object.
        """
        config1 = config_setup

        # Call it again with the same file paths using config_setup's paths
        config2 = Config(
            config_file_path=config1.experiment.config_file_path,
            project_file_path=config1.project.pyproject_file_path,
        )

        # 'is' checks for object identity, not just equality. This is the key.
        assert config1 is config2

    def test_different_params_return_new_instance(self, config_setup, tmp_path):
        """
        Calling Config() with different parameters should create a new instance.
        """
        config1 = config_setup

        # Create a completely new, different run file in a properly named folder
        exp_dir = tmp_path / "other_experiment"
        exp_dir.mkdir()
        new_run_file = exp_dir / "config.yaml"
        new_run_file.write_text("experiment_name: other_experiment\nexperiment_version: 1")

        # Initialize with the new run file
        config2 = Config(
            config_file_path=new_run_file, project_file_path=config1.project.pyproject_file_path
        )

        assert config1 is not config2
        assert config1.experiment.experiment_name == "mock_exp"
        assert config2.experiment.experiment_name == "other_experiment"
