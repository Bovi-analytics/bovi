from bovi_core.config import Config, ConfigNode

# --- TEST CLASSES ---


class ModelsConfigNode(ConfigNode):
    """A specialized class just for the 'models' section."""

    def get_model_names(self):
        return [name for name in self.__dict__ if not name.startswith("_")]


class ExperimentConfigNode(ConfigNode):
    """A specialized class for another section."""

    pass


# This is the mock version of the get_config_class function we will patch.
def mock_get_config_class(key: str):
    """
    Mock implementation of the type factory.
    Returns a special class for 'models', otherwise returns the default.
    """
    if key == "models":
        return ModelsConfigNode
    if key == "experiment_settings":
        return ExperimentConfigNode
    return ConfigNode  # Default fallback


def test_specialized_config_node_is_created(config_setup, mocker):
    """
    CRITICAL: Test that the factory correctly assigns a specialized class
    to the 'models' key in the config.
    """
    # We patch the import inside the method where it's called in the source code
    mocker.patch(
        "bovi_core.types.config_types.get_config_class",
        side_effect=mock_get_config_class,
    )

    # We need to re-initialize the config *after* the patch is in place
    from bovi_core.config import Config

    Config._instance = None
    # Use the config_file_path and pyproject_file_path from the config_setup fixture
    config = Config(
        config_file_path=config_setup.experiment.config_file_path,
        project_file_path=config_setup.project.pyproject_file_path,
    )

    # Assert that config.experiment.models is an instance of our special class
    assert isinstance(config.experiment.models, ModelsConfigNode)

    # And we can even test that it has the special methods we defined
    assert "yolo" in config.experiment.models.get_model_names()
    assert "snn" in config.experiment.models.get_model_names()


def test_default_config_node_is_used(config_setup, mocker):
    """
    Test that other keys that don't have a special type get the
    default ConfigNode class.
    """
    mocker.patch(
        "bovi_core.types.config_types.get_config_class",
        side_effect=mock_get_config_class,
    )

    Config._instance = None
    # Use the config_file_path and pyproject_file_path from the config_setup fixture
    config = Config(
        config_file_path=config_setup.experiment.config_file_path,
        project_file_path=config_setup.project.pyproject_file_path,
    )

    # The 'models' section is special
    assert isinstance(config.experiment.models, ModelsConfigNode)

    # But other sections should be standard ConfigNode
    # Test with a section that exists in the run config
    assert not isinstance(config.experiment, ModelsConfigNode)
    assert isinstance(config.experiment, ConfigNode)
