import functools
import inspect
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from azure.storage.blob import BlobServiceClient

from bovi_core.secrets import SecretsManager
from bovi_core.utils.config_utils import (
    ConfigFileTracker,
    extract_experiment_name_from_path,
    get_author_info,
    validate_project_name,
)
from bovi_core.utils.env_utils import detect_environment, get_toml_data
from bovi_core.utils.path_utils import (
    get_experiment_paths,
    get_project_paths,
    get_project_root,
    get_run_config_path,
    make_path_absolute,
)


def with_config(func):
    """
    Decorator to inject the Config singleton into a function's kwargs.
    If the function signature includes 'config' and it is not provided in the call,
    this decorator initializes a default Config() instance and passes it.
    """
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if 'config' is a parameter in the decorated function
        if "config" in sig.parameters:
            # Check if 'config' was already provided, either positionally or by keyword
            bound_args = sig.bind_partial(*args, **kwargs).arguments
            if "config" not in bound_args:
                # If not provided, inject the default Config instance
                kwargs["config"] = Config()

        return func(*args, **kwargs)

    return wrapper


class ConfigNode:
    """A simple nested object for accessing configuration data via attributes."""

    # Type hints for dynamically added attributes
    name: str  # Guaranteed non-None after initialization
    src_dir: str  # Guaranteed non-None after initialization
    notebooks_dir: str  # Guaranteed non-None after initialization
    data_dir: str  # Guaranteed non-None after initialization

    def __init__(self, data, secrets_manager=None, is_secrets=False, mutable_keys=None):
        self._secrets_manager = secrets_manager
        self._is_secrets = is_secrets
        self._mutable_keys = mutable_keys or set()
        self._secret_keys = {}  # Store secret key mappings without exposing them

        for key, value in data.items():
            new_path_is_secrets = is_secrets or key == "secrets"
            if isinstance(value, dict):
                # Use the factory method to create the right type
                setattr(self, key, self._create_config_node(key, value, new_path_is_secrets))
            elif new_path_is_secrets and self._secrets_manager and key != "secrets":
                # For secrets, store the mapping but don't create visible properties
                self._secret_keys[key] = value
            else:
                setattr(self, key, value)

    def _create_config_node(self, key: str, data: dict, is_secrets: bool):
        """Factory method to create the appropriate ConfigNode subclass."""
        # Import here to avoid circular dependency
        from .types.config_types import get_config_class

        config_class = get_config_class(key)
        return config_class(data, self._secrets_manager, is_secrets, self._mutable_keys)

    def list_keys(self, prefix=""):
        keys = []
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, ConfigNode):
                keys.extend(v.list_keys(full_key))
            else:
                keys.append(full_key)
        return keys

    def __setattr__(self, name, value):
        # Only allow mutation for mutable_keys (run config keys)
        if (
            hasattr(self, "_mutable_keys")
            and name not in getattr(self, "_mutable_keys", set())
            and not name.startswith("_")
        ):
            if hasattr(self, name):
                raise ValueError(
                    f"Cannot update '{name}' - this is a project configuration key "
                    f"and is read-only."
                )
        super().__setattr__(name, value)

    def __getattr__(self, name: str):
        # Check if this is a secret key request (use __dict__ to avoid recursion)
        secret_keys = self.__dict__.get("_secret_keys", {})
        if secret_keys and name in secret_keys:
            secrets_manager = self.__dict__.get("_secrets_manager")
            if secrets_manager:
                return secrets_manager.get_secret(secret_keys[name])
            else:
                raise RuntimeError("SecretsManager not available")

        available_keys = [k for k in self.__dict__.keys() if not k.startswith("_")]

        # Create a helpful, context-aware error message.
        error_msg = f"Attribute '{name}' not found on this ConfigNode. "
        if available_keys:
            error_msg += f"Available attributes are: {', '.join(sorted(available_keys))}"
        else:
            error_msg += "This node has no attributes."

        raise AttributeError(error_msg)


class Config:
    _instance = None
    _instance_params = None
    _blob_service_client = None  # Shared blob service client for connection pooling
    _file_tracker: Optional[ConfigFileTracker] = None  # Tracks config file changes

    # Instance attributes (set in _initialize)
    experiment: ConfigNode  # Experiment config (from config.yaml)

    def __new__(
        cls,
        experiment_name: Optional[str] = None,
        config_file_name: str = "config.yaml",
        config_file_path: Optional[str] = None,
        project_file_path: Optional[str] = None,
        project_name: Optional[str] = None,
    ):
        current_params = (
            experiment_name,
            config_file_name,
            config_file_path,
            project_file_path,
            project_name,
        )

        # Check if tracked config files have changed - invalidate singleton if so
        if cls._instance is not None and cls._file_tracker is not None:
            if cls._file_tracker.any_changed():
                print("🔄 Config files changed, reloading...")
                cls._instance = None
                cls._file_tracker = None

        # If no params provided (or only default config_file_name) and instance exists,
        # return existing instance regardless of how it was originally created
        if experiment_name is None and config_file_path is None and cls._instance is not None:
            return cls._instance

        # Otherwise, check for exact match or create new
        if cls._instance is None or cls._instance_params != current_params:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance_params = current_params
            cls._instance._initialize(
                experiment_name, config_file_name, config_file_path, project_file_path, project_name
            )
        return cls._instance

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        config_file_name: str = "config.yaml",
        config_file_path: Optional[str] = None,
        project_file_path: Optional[str] = None,
        project_name: Optional[str] = None,
    ):
        """Initialize a Config instance for experiment management.

        Args:
            experiment_name: Name of the experiment. If None, will be extracted
                from config_file_path.
            config_file_name: Name of the config file (default: "config.yaml").
            config_file_path: Path to the experiment config file. If None, will be
                auto-generated from experiment_name.
            project_file_path: Path to pyproject.toml. If None, will be auto-detected.
            project_name: Name of the project (must match ``project.name`` in
                pyproject.toml). Used to locate the correct pyproject.toml in
                monorepo layouts. If None, the first pyproject.toml found is used.

        Note:
            Either experiment_name OR config_file_path must be provided. The Config class
            uses singleton pattern,
            so multiple calls with the same parameters return the same instance.
        """

        # __init__ is called every time, but we only want to set attributes if this
        # is a new instance
        pass

    def _initialize(self, experiment_name, config_file_name, config_file_path, project_file_path, project_name=None):
        print("🔧 Initializing Config...")

        # Initialize file tracker for change detection
        Config._file_tracker = ConfigFileTracker()

        self.environment = detect_environment()
        print(f"   🌍 Environment: {self.environment}")

        if experiment_name is None:
            if config_file_path is None:
                raise ValueError("experiment_name or config_file_path must be provided")
            else:
                print(
                    f"🔍 No experiment_name passed, resolving from config_file_path: {config_file_path}"
                )
                experiment_name = extract_experiment_name_from_path(config_file_path)
                if experiment_name is None:
                    # Fallback to parent folder name if not in experiments structure
                    experiment_name = Path(config_file_path).parent.name
                print(f"      🧪 Resolved experiment_name: {experiment_name}")

        self.experiment_name = experiment_name
        self.config_file_name = config_file_name

        if project_file_path is None:
            print("🔍 No TOML path passed, resolving path automatically")
            project_root_path = get_project_root(project_name=project_name)
            project_file_path = os.path.join(project_root_path, "pyproject.toml")
        else:
            project_root_path = Path(project_file_path).parent

        self.project_file_path = project_file_path
        Config._file_tracker.track_file(project_file_path)  # Track pyproject.toml changes
        project_data = get_toml_data(project_file_path)

        project_data_clean = {}
        for k, v in project_data.items():
            if k in ["tool", "project"]:
                project_data_clean.update(v)
            else:
                project_data_clean[k] = v
        self.project = ConfigNode(project_data_clean)

        # Validate project name is not template default
        validate_project_name(self.project.name, str(project_root_path))

        project_info = {
            "project": {
                "name": self.project.name,
                "workspace_user": getattr(self.project, "workspace_user", "shared"),
            }
        }

        paths = get_project_paths(self.environment, str(project_root_path), project_info)
        # Set project paths as regular attributes
        self.project.project_root = paths["project_root_path"]
        self.project.pyproject_file_path = project_file_path
        self.project.src_dir = paths["src_dir_path"]
        self.project.notebooks_dir = paths["notebooks_dir_path"]
        self.project.data_dir = paths["data_dir_path"]

        # Validate critical attributes are resolved (for type narrowing)
        assert self.project.name is not None, "Project name not resolved from pyproject.toml"
        assert self.project.src_dir is not None, "Source directory not resolved"
        assert self.project.notebooks_dir is not None, "Notebooks directory not resolved"
        assert self.project.data_dir is not None, "Data directory not resolved"

        # Store raw secrets mapping for SecretsManager access (avoid circular dependency)
        self._raw_secrets_mapping = project_data_clean.get("secrets", {})

        # Setup secrets manager
        self.secrets_manager = self._setup_secrets_manager()

        # Recreate project ConfigNode with secrets_manager for proper secret property handling
        self.project = ConfigNode(project_data_clean, self.secrets_manager)

        # Reset project paths as regular attributes after recreation
        self.project.project_root = paths["project_root_path"]
        self.project.pyproject_file_path = project_file_path
        self.project.src_dir = paths["src_dir_path"]
        self.project.notebooks_dir = paths["notebooks_dir_path"]
        self.project.data_dir = paths["data_dir_path"]

        # Extract and validate author info (raises AuthorConfigError if not configured)
        self.author_name, self.author_email = get_author_info(self.project)
        print(f"   👤 Author: {self.author_name} <{self.author_email}>")

        if config_file_path is None:
            print("🔍 No config_file_path passed, resolving from:")
            print(f"      📁 project_root_path: {project_root_path}")
            print(f"      🧪 experiment_name: {experiment_name}")
            print(f"      📄 config_file_name: {config_file_name}")
            config_file_path = get_run_config_path(
                str(project_root_path), experiment_name, config_file_name
            )
            print(f"      📄 Resolved config_file_path: {config_file_path}")

        project_vars = self._flatten_config_node(self.project)

        # Load YAML (run config)
        if config_file_path:
            if not os.path.exists(config_file_path):
                experiments_dir = os.path.join(project_root_path, "data", "experiments")
                # Look for config files in the versioned config directory (parent of config_file_path)
                config_dir = str(Path(config_file_path).parent)
                available_configs = (
                    self._list_available_configs(config_dir)
                    if os.path.exists(config_dir)
                    else "Config directory does not exist"
                )
                raise FileNotFoundError(
                    f"❌ Run config file not found for '{experiment_name}':\n"
                    f"   Expected path: {config_file_path}\n"
                    f"   Looking for: {config_file_name}\n\n"
                    f"   Make sure the directory structure exists:\n"
                    f"   data/experiments/{experiment_name}/{config_file_name}\n\n"
                    f"   Available config files in '{experiment_name}': {available_configs}\n"
                    f"   Available experiments: {self._list_available_experiments(experiments_dir) if os.path.exists(experiments_dir) else 'None found'}"
                )
            try:
                with open(config_file_path, "r") as f:
                    print(f"📄 Loading run config file: {config_file_path}")
                    run_data = yaml.safe_load(f)
                    if run_data is None:
                        raise ValueError("Config file is empty or invalid YAML")
                    Config._file_tracker.track_file(config_file_path)  # Track run config changes
            except FileNotFoundError:
                raise
            except ValueError:
                raise
            except Exception as e:
                raise RuntimeError(f"Error loading run config file: {e}")

            # Validate required fields in config
            self._validate_run_config(run_data, config_file_path, experiment_name)

            # Resolve relative paths to absolute paths
            run_data = self._resolve_paths(run_data, Path(project_root_path))

            # --- Main Templating Logic ---
            path_templates = run_data.pop("path_templates", {})
            experiment_vars = {
                "experiment_version": run_data.get("experiment_version", "1"),
                "experiment_name": run_data.get("experiment_name", ""),
            }
            if "models" in run_data:
                processed_models = self._process_templated_models(
                    run_data.get("models", {}),
                    path_templates,
                    project_vars,
                    experiment_vars,
                    project_root=Path(project_root_path),
                )
                run_data["models"] = processed_models

            # Store path_templates for potential re-processing (useful for testing)
            self._path_templates = path_templates

            mutable_keys = set(run_data.keys())
            self.experiment = ConfigNode(
                run_data,
                secrets_manager=self.secrets_manager,
                mutable_keys=mutable_keys,
            )

            # Add experiment paths as attributes on config.experiment
            exp_version = run_data.get("experiment_version", "1")
            exp_paths = get_experiment_paths(
                str(project_root_path),
                experiment_name,
                str(exp_version),
            )
            self.experiment.experiments_dir = exp_paths["experiments_dir"]
            self.experiment.dir = exp_paths["dir"]
            self.experiment.config_dir = exp_paths["config_dir"]
            self.experiment.weights_dir = exp_paths["weights_dir"]
            self.experiment.config_file_path = config_file_path
        else:
            raise RuntimeError("Could not resolve the run file path. Config Not set correctly.")

        # Setup secrets as attribute for direct access
        if hasattr(self.project, "secrets"):
            # Ensure project.secrets has the secrets_manager
            self.project.secrets._secrets_manager = self.secrets_manager
            self.secrets = self.project.secrets
        else:
            # Create empty secrets node if none exists
            self.secrets = ConfigNode({}, self.secrets_manager, is_secrets=True)

        # Setup unity catalog environment variable
        if "project_vars" in locals():
            if "databricks_workspace_host" in locals()["project_vars"]:
                os.environ["DATABRICKS_HOST"] = locals()["project_vars"][
                    "databricks_workspace_host"
                ]

        print("✅ Config initialization complete!")

    def _setup_secrets_manager(self):
        try:
            if self.environment == "databricks" and not hasattr(self.project, "environments"):
                print("⚠️ Running on Spark worker - skipping secrets manager initialization")
                return None
            else:
                manager = SecretsManager(self.environment, self)
                manager.populate_environment()
                return manager
        except Exception as e:
            print(f"⚠️ Could not initialize secrets manager: {e}")
            return None

    def _list_available_experiments(self, experiments_dir: str) -> str:
        """List available experiment directories."""
        try:
            if not os.path.exists(experiments_dir):
                return "Directory does not exist"
            experiments = [
                d
                for d in os.listdir(experiments_dir)
                if os.path.isdir(os.path.join(experiments_dir, d))
            ]
            if not experiments:
                return "No experiments found"
            return ", ".join(sorted(experiments))
        except Exception:
            return "Unable to list experiments"

    def _list_available_configs(self, experiment_folder: str) -> str:
        """List available config files (.yaml, .yml) in an experiment folder."""
        try:
            if not os.path.exists(experiment_folder):
                return "Experiment folder does not exist"
            config_files = [
                f for f in os.listdir(experiment_folder) if f.endswith((".yaml", ".yml"))
            ]
            if not config_files:
                return "No config files found"
            return ", ".join(sorted(config_files))
        except Exception:
            return "Unable to list config files"

    def _validate_run_config(self, run_data: dict, config_file_path: str, experiment_name: str):
        """Validate that the loaded run config has all required fields."""
        if run_data is None:
            raise ValueError(f"Config file is empty or invalid YAML: {config_file_path}")

        required_fields = ["experiment_name", "experiment_version"]
        missing_fields = [f for f in required_fields if f not in run_data]

        if missing_fields:
            raise ValueError(
                f"❌ Invalid run config '{config_file_path}':\n"
                f"   Missing required fields: {', '.join(missing_fields)}\n"
                f"   The config file must define these at the top level."
            )

        # Validate experiment name matches the folder name (source of truth)
        folder_name = extract_experiment_name_from_path(config_file_path)
        if folder_name is None:
            # Fallback to parent folder name if not in experiments structure
            folder_name = Path(config_file_path).parent.name
        config_exp_name = run_data.get("experiment_name")

        if config_exp_name != folder_name:
            raise ValueError(
                f"❌ Experiment name mismatch:\n"
                f"   Folder name: '{folder_name}'\n"
                f"   experiment_name in config: '{config_exp_name}'\n"
                f"   File: {config_file_path}\n\n"
                f"   The 'experiment_name' field in your config.yaml must match the folder name.\n"
                f"   Either:\n"
                f"   1. Rename the folder to: data/experiments/{config_exp_name}/\n"
                f"   2. Update experiment_name in config.yaml to: '{folder_name}'\n"
            )

    def _resolve_paths(self, data: dict, project_root: Path) -> dict:
        """
        Recursively resolve relative paths in config data to absolute paths.

        Any string value in a key ending with '_path' or '_dir' that is a relative
        path will be resolved relative to the project root.

        Args:
            data: Config dictionary (possibly nested)
            project_root: Project root path for resolving relative paths

        Returns:
            Config dictionary with resolved paths
        """
        if not isinstance(data, dict):
            return data

        resolved = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively resolve nested dicts
                resolved[key] = self._resolve_paths(value, project_root)
            elif isinstance(value, list):
                # Handle lists (e.g., transforms list)
                resolved[key] = [
                    self._resolve_paths(item, project_root) if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, str) and (key.endswith("_path") or key.endswith("_dir")):
                # Resolve path-like fields
                resolved[key] = str(make_path_absolute(value, project_root))
            else:
                resolved[key] = value

        return resolved

    def _flatten_config_node(self, node, parent_key="", sep="_"):
        """Flattens a ConfigNode into a single-level dictionary."""
        items = {}
        for k, v in node.__dict__.items():
            if k.startswith("_"):
                continue
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, ConfigNode):
                items.update(self._flatten_config_node(v, new_key, sep=sep))
            else:
                # Use project.name instead of name to avoid ambiguity
                if new_key == "name" and hasattr(self.project, "name"):
                    items["project_name"] = v
                else:
                    items[new_key] = v
        return items

    def _process_templated_models(
        self,
        models_data,
        path_templates,
        project_vars,
        experiment_vars=None,
        project_root=None,
    ):
        """Processes model configurations to generate paths from templates.

        Args:
            models_data: Dictionary of model configurations
            path_templates: Dictionary of path template definitions
            project_vars: Dictionary of project-level variables (from pyproject.toml)
            experiment_vars: Dictionary of experiment-level variables (from config.yaml)
            project_root: Path to project root for resolving relative paths

        Returns:
            Dictionary of processed model configurations with resolved paths

        """
        if experiment_vars is None:
            experiment_vars = {}

        processed_models = {}
        for model_name, model_config in models_data.items():
            # Start with the original model configuration
            processed_model = model_config.copy()

            # Prepare context for template substitution
            # Order: project_vars < experiment_vars < model-specific vars
            context = project_vars.copy()
            context.update(experiment_vars)
            context["model_name"] = model_config.get("vars", {}).get("model_name", model_name)
            context.update(model_config.get("vars", {}))
            if "version" in model_config:
                context["model_version"] = model_config["version"]

            if "template_vars" in model_config:
                # Iterate through the data sources (e.g., 'weights_file', 'config_file')
                for placeholder_key, source_data in model_config["template_vars"].items():
                    # Find all path templates that use this data source
                    for template_name, template_config in path_templates.items():
                        if template_config.get("uses") == placeholder_key:
                            # Create the nested dictionary for the output (e.g., 'weights_blob')
                            processed_model.setdefault(template_name, {})
                            # Generate a path for each item in the source data
                            for name, value in source_data.items():
                                final_context = context.copy()
                                final_context[placeholder_key] = value

                                # Use str.format() instead of Template.safe_substitute() since
                                # our templates use {var} format
                                try:
                                    resolved_path = template_config["template"].format(
                                        **final_context
                                    )
                                except KeyError:
                                    # If a variable is missing, use safe_substitute behavior -
                                    # leave the placeholder
                                    resolved_path = template_config["template"]
                                    for key, value in final_context.items():
                                        resolved_path = resolved_path.replace(
                                            f"{{{key}}}", str(value)
                                        )
                                # Resolve to absolute path if project_root provided
                                if project_root is not None:
                                    resolved_path = str(
                                        make_path_absolute(resolved_path, project_root)
                                    )
                                processed_model[template_name][name] = resolved_path

            # Clean up by removing the template instructions from the final object
            processed_model.pop("template_vars", None)
            processed_models[model_name] = processed_model

        return processed_models

    @property
    @lru_cache(maxsize=None)  # Cache the client instance
    def container_client(self):
        """
        Initializes and returns a Blob Storage container client.
        The client is cached for the lifetime of the Config instance.
        """
        if self.experiment and self.experiment.verbose > 0:
            print("🔧 Initializing Blob Storage Container Client...")
        try:
            storage_account_name = self.project.blob_storage.storage_account_name
            container_name = self.project.blob_storage.container_name
        except AttributeError as e:
            raise ValueError(f"Missing blob storage configuration in project settings: {e}")

        storage_account_key = self.secrets.storage_account_key

        if not storage_account_name:
            raise ValueError("storage_account_name not found in config.project.blob_storage")
        if not storage_account_key:
            raise ValueError(
                "storage_account_key not found in config.secrets (check Databricks secrets)"
            )
        if not container_name:
            raise ValueError("container_name not found in config.project.blob_storage")

        connect_str = (
            f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};"
            f"AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
        )

        try:
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            print(f"   Retrieved connection from string with account {storage_account_name}")
        except Exception as e:
            print(f"❌ Could not retrieve connection from string: {e}")
            raise e

        try:
            client = blob_service_client.get_container_client(container_name)
            if self.experiment and self.experiment.verbose > 0:
                print(f"   Retrieved container client for {container_name}")
                print("✅ Blob client initialization complete!")
            return client
        except Exception as e:
            print(f"❌ Could not retrieve container client: {e}")
            raise e

    @lru_cache(maxsize=4)
    def get_blob_container_client(self, container_name: Optional[str] = None):
        """
        Get cached blob container client with connection pooling.

        This method provides a shared BlobServiceClient instance that can be reused
        across DataLoader workers for improved performance.

        Args:
            container_name: Container name (if None, uses default from config)

        Returns:
            ContainerClient instance
        """
        if container_name is None:
            container_name = self.project.blob_storage.container_name

        if not container_name:
            raise ValueError("container_name not found in config.project.blob_storage")

        # Initialize shared blob service client if needed
        if self._blob_service_client is None:
            storage_account_name = self.project.blob_storage.storage_account_name
            storage_account_key = self.secrets.storage_account_key

            if not storage_account_name:
                raise ValueError("storage_account_name not found in config.project.blob_storage")
            if not storage_account_key:
                raise ValueError(
                    "storage_account_key not found in config.secrets (check Databricks secrets)"
                )

            connect_str = (
                f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};"
                f"AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
            )

            self._blob_service_client = BlobServiceClient.from_connection_string(
                connect_str,
                connection_timeout=60,
                max_single_get_size=4 * 1024 * 1024,  # 4MB chunks
                max_chunk_get_size=4 * 1024 * 1024,
            )

        return self._blob_service_client.get_container_client(container_name)

    def list_keys(self):
        return {
            "project_keys": self.project.list_keys() if self.project else [],
            "experiment_keys": self.experiment.list_keys() if self.experiment else [],
            "secret_keys": self.secrets.list_keys() if self.secrets else [],
            "environment_info": {"environment": self.environment},
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("secrets_manager", None)
        # Don't serialize the cached client property
        if "_container_client" in state:
            del state["_container_client"]
        # Don't serialize blob service client (will recreate on worker)
        if "_blob_service_client" in state:
            state["_blob_service_client"] = None
        # Clear lru_cache on serialization to avoid serialization issues
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-initialize secrets manager on the worker if needed, but safer to leave it as None
        self.secrets_manager = None
        Config._instance = self
        Config._instance_params = (None, None)

    @classmethod
    def reset(cls) -> None:
        """
        Force reset the Config singleton.

        Clears the cached instance, allowing a fresh Config to be created
        on next instantiation. Useful for:
        - Testing (resetting state between tests)
        - Notebooks (forcing reload after manual file edits)
        - Development (when auto-detection doesn't catch changes)

        Example:
            >>> Config.reset()
            >>> config = Config(experiment_name="my_experiment")  # Fresh instance
        """
        cls._instance = None
        cls._instance_params = None
        cls._file_tracker = None
        cls._blob_service_client = None
        print("🔄 Config singleton reset")


if __name__ == "__main__":
    print("🔧 Testing Config system...")
