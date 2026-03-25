from typing import Dict, Optional, Type

from bovi_core.config import ConfigNode


class BlobStorageConfig(ConfigNode):
    storage_account_name: str
    container_name: str


class DatabricksConfig(ConfigNode):
    cluster_name: str
    workspace_url: str


class EnvironmentConfig(ConfigNode):
    databricks: "DatabricksConfig"
    local: "DatabricksConfig"


class ProjectConfig(ConfigNode):
    project_root: Optional[str]
    project_file_path: Optional[str]
    src_dir: Optional[str]
    notebooks_dir: Optional[str]
    blob_storage: BlobStorageConfig
    databricks: DatabricksConfig
    environments: EnvironmentConfig


# Registry of config classes
_CONFIG_CLASSES: Dict[str, Type[ConfigNode]] = {
    "blob_storage": BlobStorageConfig,
    "databricks": DatabricksConfig,
    "environments": EnvironmentConfig,
}


def get_config_class(key: str) -> Type[ConfigNode]:
    """Get the appropriate ConfigNode subclass for a given key."""
    return _CONFIG_CLASSES.get(key, ConfigNode)
