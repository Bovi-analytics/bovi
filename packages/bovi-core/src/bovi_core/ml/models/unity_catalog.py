"""Unity Catalog registration utilities for ML models."""

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from bovi_core.config import Config
    from bovi_core.ml.dataloaders.base import Dataset
    from bovi_core.ml.models.model import Model

logger = logging.getLogger(__name__)


def _get_next_experiment_run(
    experiment_base_path: str,
    verbose: bool = False,
) -> int:
    """
    Determine the next experiment run number by checking existing MLflow experiments.

    Looks for experiments matching the pattern {experiment_base_path}/run_{n}
    and returns the next available run number.

    Args:
        experiment_base_path: Base path without run number
            (e.g., "/Users/email@example.com/projects/
            bovi-models-template/.../lactation_autoencoder/versions/1")
        verbose: Print debug information

    Returns:
        Next available run number (0 if no existing runs)
    """
    import mlflow

    # Search for existing experiments with this base path
    try:
        # List all experiments and filter by prefix
        experiments = mlflow.search_experiments(
            filter_string=f"name LIKE '{experiment_base_path}/run_%'"
        )

        if not experiments:
            return 0

        # Extract run numbers from experiment names
        run_numbers = []
        for exp in experiments:
            # Extract run number from path like ".../run_0", ".../run_1"
            name = exp.name
            if "/run_" in name:
                try:
                    run_num = int(name.split("/run_")[-1])
                    run_numbers.append(run_num)
                except ValueError:
                    continue

        if not run_numbers:
            return 0

        next_run = max(run_numbers) + 1
        if verbose:
            print(f"   🔢 Found existing runs: {sorted(run_numbers)}, using run_{next_run}")
        return next_run

    except Exception as e:
        if verbose:
            print(f"   ⚠️  Could not check existing runs: {e}, starting at run_0")
        return 0


def _generate_mlflow_experiment_name(
    config: "Config",
    verbose: bool = False,
) -> str:
    """
    Generate MLflow experiment name from config.

    Format: /Users/{email}/projects/{project_name}/data/
    experiments/{experiment_name}/versions/{version}/run_{n}

    Args:
        config: Config instance with project and run settings (must have author_name, author_email)
        verbose: Print debug information

    Returns:
        Full MLflow experiment path

    Note:
        Author info is validated during Config initialization. If config was created
        successfully, author_name and author_email are guaranteed to be valid.
    """
    # Author info is already validated and stored in config during initialization
    author_name = config.author_name
    author_email = config.author_email

    # Get project name from config
    project_name = config.project.name

    # Get experiment info from config
    experiment_name = getattr(config.experiment, "experiment_name", "default_experiment")
    experiment_version = getattr(config.experiment, "experiment_version", "v1")

    # Build base path (without run number)
    # Format: /Users/{email}/projects/{project_name}/data/
    # experiments/{experiment_name}/versions/{version}
    experiment_base_path = (
        f"/Users/{author_email}/projects/{project_name}"
        f"/data/experiments/{experiment_name}"
        f"/versions/{experiment_version}"
    )

    # Get next run number
    run_number = _get_next_experiment_run(experiment_base_path, verbose=verbose)

    # Full experiment path
    full_path = f"{experiment_base_path}/run_{run_number}"

    if verbose:
        print(f"   👤 Author: {author_name} <{author_email}>")
        print(f"   📊 MLflow experiment: {full_path}")

    return full_path


def _ensure_workspace_path_exists(path: str, verbose: bool = False) -> None:
    """
    Create parent directories in Databricks Workspace if they don't exist.

    Uses Databricks Workspace API to create folder hierarchy.

    Args:
        path: Workspace path to ensure exists (e.g., "/Users/email/projects/exp/versions/1")
        verbose: Print debug information
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    # Split path into components and create each level
    parts = path.strip("/").split("/")
    current_path = ""

    for part in parts:
        current_path = f"{current_path}/{part}"
        try:
            w.workspace.get_status(current_path)
        except Exception:
            # Directory doesn't exist, create it
            if verbose:
                print(f"   📁 Creating directory: {current_path}")
            w.workspace.mkdirs(current_path)


def _cleanup_experiment_artifacts(experiment_path: str, verbose: bool = False) -> None:
    """
    Remove staging experiment and artifacts after successful UC registration.

    Args:
        experiment_path: Full experiment path to clean up
        verbose: Print debug information
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    try:
        if verbose:
            print(f"   🧹 Cleaning up staging: {experiment_path}")
        w.workspace.delete(experiment_path, recursive=True)
    except Exception as e:
        if verbose:
            print(f"   ⚠️  Cleanup failed (non-critical): {e}")


def register_to_unity_catalog(
    model: "Model",
    catalog: str = "projects",
    schema: str = "bovi_core",
    model_name: Optional[str] = None,
    alias: Optional[str] = None,
    dataset: Optional["Dataset"] = None,
    input_example: Optional[Any] = None,
    signature: Optional[Any] = None,
    description: Optional[str] = None,
    tags: Optional[dict] = None,
    n_samples: int = 5,
    auto_increment_version: bool = True,
    verbose: bool = True,
    pyfunc_wrapper_class: Optional[type] = None,
    mlflow_experiment_name: Optional[str] = None,
):
    """
    Register model to Unity Catalog with auto-generated signature.

    This method handles the complete workflow for registering a model
    to Databricks Unity Catalog:
    1. Generate model name (if not provided)
    2. Auto-generate signature from dataset (if not provided)
    3. Create input example (if not provided)
    4. Log model to MLflow
    5. Register to Unity Catalog
    6. Set alias and metadata

    Args:
        model: Model instance to register
        catalog: Unity Catalog catalog name (default: "projects")
        schema: Schema within catalog (default: "bovi_core")
        model_name: Model name (auto-generated from config if None)
        alias: Model alias/version tag (e.g., "Champion", "v1.0")
        dataset: Dataset to generate signature from (recommended)
        input_example: Manual input example (overrides dataset)
        signature: Manual MLflow signature (overrides auto-inference)
        description: Model description for documentation
        tags: Additional tags (merged with auto-generated tags)
        n_samples: Number of samples for signature inference
        auto_increment_version: Auto-increment experiment version if alias exists
        verbose: Print registration progress
        pyfunc_wrapper_class: Optional pyfunc wrapper class for TensorFlow SavedModels.
            Required for TF SavedModels to map semantic names to generic TF names.
        mlflow_experiment_name: MLflow experiment path for Databricks tracking.
            If None, auto-generated as "/Users/{email}/projects/
            {project_name}/.../versions/{version}/run_{n}"
            where email and project_name come from config, and run number auto-increments.
            Parent directories are created automatically, and staging artifacts are cleaned up
            after successful registration to Unity Catalog.

    Returns:
        mlflow.entities.model_registry.ModelVersion

    Example:
        >>> # Auto-generate everything from dataset
        >>> from bovi_core.ml.models.unity_catalog import register_to_unity_catalog
        >>> register_to_unity_catalog(
        ...     model=model,
        ...     dataset=train_dataset,
        ...     alias="Champion",
        ...     description="YOLO model for cow detection"
        ... )

        >>> # Manual signature (advanced)
        >>> from mlflow.models import infer_signature
        >>> input_ex = {"image": np.zeros((1, 3, 640, 640))}
        >>> sig = infer_signature(input_ex)
        >>> register_to_unity_catalog(
        ...     model=model,
        ...     input_example=input_ex,
        ...     signature=sig
        ... )

        >>> # Custom catalog and schema
        >>> register_to_unity_catalog(
        ...     model=model,
        ...     catalog="production",
        ...     schema="models",
        ...     model_name="cow_detector_yolo",
        ...     dataset=dataset
        ... )

    Raises:
        ImportError: If mlflow is not installed
        ValueError: If neither dataset nor input_example is provided
    """
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError:
        raise ImportError("mlflow is required for Unity Catalog registration. ")

    # Ensure Unity Catalog registry
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    if verbose:
        print("🔧 Starting Unity Catalog model registration...")

    # Generate or use provided MLflow experiment name
    if mlflow_experiment_name is None:
        mlflow_experiment_name = _generate_mlflow_experiment_name(
            config=model.config,
            verbose=verbose,
        )
    elif verbose:
        print(f"   📊 MLflow experiment: {mlflow_experiment_name}")

    # Ensure parent directories exist in Databricks Workspace
    # Extract parent path (everything before /run_N)
    parent_path = mlflow_experiment_name.rsplit("/", 1)[0]
    _ensure_workspace_path_exists(parent_path, verbose=verbose)

    # Set the experiment before starting any runs
    mlflow.set_experiment(mlflow_experiment_name)

    # 1. Generate model name
    full_model_name = _generate_uc_model_name(model, catalog, schema, model_name)
    if verbose:
        print(f"   📝 Model name: {full_model_name}")

    # 2. Generate or validate alias
    if alias and auto_increment_version:
        alias = _resolve_alias_version(full_model_name, alias, verbose=verbose)

    # 3. Generate signature and input example
    if signature is None or input_example is None:
        if dataset is None and input_example is None:
            raise ValueError(
                "Must provide either 'dataset' or 'input_example' for signature generation"
            )

        if dataset is not None:
            # Get raw input from dataset
            if input_example is None:
                import numpy as np

                raw_input = dataset.get_input_example(n_samples=1, batch=True)

                # Transform through predictor to get TensorFlow-ready shapes
                if hasattr(model, "predictor") and hasattr(model.predictor, "_prepare_inputs"):
                    # Convert batched dict to list for predictor
                    batch_size = next(iter(raw_input.values())).shape[0]
                    data_list = [{k: v[i] for k, v in raw_input.items()} for i in range(batch_size)]
                    # Apply transforms - this may return TensorFlow tensors
                    transformed_inputs = model.predictor._prepare_inputs(data_list)

                    # Convert TensorFlow tensors to numpy arrays for MLflow compatibility
                    # MLflow requires numpy arrays (JSON serializable) for input_example
                    input_example = {}
                    for k, v in transformed_inputs.items():
                        if hasattr(v, "numpy"):
                            # TensorFlow tensor - convert to numpy
                            input_example[k] = v.numpy()
                        elif isinstance(v, np.ndarray):
                            input_example[k] = v
                        else:
                            # Try to convert to numpy array
                            input_example[k] = np.asarray(v)
                else:
                    input_example = raw_input

            # Generate signature from transformed inputs
            if signature is None:
                if verbose:
                    print("   🔍 Generating signature from transformed inputs...")

                # Call model directly with transformed inputs
                # (skip predictor to avoid double transform)
                import tensorflow as tf

                # Convert numpy arrays to TensorFlow tensors for model inference
                tf_inputs = {k: tf.constant(v, dtype=tf.float32) for k, v in input_example.items()}
                raw_output = model(**tf_inputs)

                # Convert to base format
                from bovi_core.ml.utils.signature_utils import output_to_serializable

                if isinstance(raw_output, dict):
                    predictions = {k: v.numpy() for k, v in raw_output.items()}
                else:
                    predictions = raw_output.numpy()
                predictions = output_to_serializable(predictions)

                from mlflow.models import infer_signature

                signature = infer_signature(input_example, predictions)

    # DEBUG: Print input example shapes
    if verbose and input_example is not None:
        print("   🔍 Input example shapes:")
        for key, value in input_example.items():
            import numpy as np

            if isinstance(value, np.ndarray):
                print(f"      {key}: {value.shape} ({value.dtype})")
            else:
                print(f"      {key}: {type(value)} = {value}")

    # 4. Build tags
    model_tags = _generate_model_tags(model, tags)

    # 5. Log and register model
    if verbose:
        print("   📦 Logging model to MLflow...")

    model_version = _log_and_register_model(
        model=model,
        full_model_name=full_model_name,
        signature=signature,
        input_example=input_example,
        tags=model_tags,
        verbose=verbose,
        pyfunc_wrapper_class=pyfunc_wrapper_class,
    )

    if verbose:
        print(f"   ✅ Registered model version: {model_version.version}")

    # 6. Set alias
    if alias:
        client = MlflowClient()
        client.set_registered_model_alias(
            name=full_model_name,
            alias=alias,
            version=model_version.version,
        )
        if verbose:
            print(f"   🏷️  Set alias: {alias}")

    # 7. Set description
    if description:
        client = MlflowClient()
        client.update_model_version(
            name=full_model_name,
            version=model_version.version,
            description=description,
        )
        if verbose:
            print("   📄 Set description")

    # Cleanup staging experiment artifacts (they're now in Unity Catalog)
    # TODO: Re-enable after investigating what artifacts remain
    # _cleanup_experiment_artifacts(mlflow_experiment_name, verbose=verbose)

    if verbose:
        print("   🎉 Registration complete!")
        print(f"   🔗 Model URI: models:/{full_model_name}@{alias}")

    return model_version


def _generate_uc_model_name(
    model: "Model", catalog: str, schema: str, model_name: Optional[str]
) -> str:
    """
    Generate full Unity Catalog model name.

    Format: {catalog}.{schema}.{model_name}

    Args:
        model: Model instance
        catalog: UC catalog name
        schema: UC schema name
        model_name: Model name (auto-generated if None)

    Returns:
        Full UC model name (e.g., "projects.bovi_core.yolo_cow_detector")
    """
    if model_name is None:
        # Auto-generate from config
        project_name = model.config.project.name
        model_type = getattr(model, "model_name", "model")
        model_name = f"{project_name}_{model_type}"

    return f"{catalog}.{schema}.{model_name}"


def _resolve_alias_version(full_model_name: str, alias: str, verbose: bool = False) -> str:
    """
    Auto-increment alias version if it already exists.

    If alias="v1.0" exists, returns "v1.1", etc.

    Args:
        full_model_name: Full UC model name
        alias: Desired alias
        verbose: Print version resolution

    Returns:
        Resolved alias (may be incremented)
    """
    from mlflow import MlflowClient

    client = MlflowClient()

    try:
        # Check if model exists
        versions = client.search_model_versions(filter_string=f"name='{full_model_name}'")

        # Find existing aliases
        existing_aliases = set()
        for version in versions:
            if hasattr(version, "aliases") and version.aliases:
                existing_aliases.update(version.aliases)

        # Check if our alias exists
        if alias not in existing_aliases:
            return alias

        # Auto-increment version
        # Try to extract version number (e.g., "v1.0" → 1.0)
        import re

        match = re.search(r"v?(\d+)\.(\d+)", alias)

        if match:
            major = int(match.group(1))
            minor = int(match.group(2))

            # Increment minor version
            new_alias = f"v{major}.{minor + 1}"

            # Check if new alias exists (recursive)
            if new_alias in existing_aliases:
                return _resolve_alias_version(full_model_name, new_alias, verbose)

            if verbose:
                print(f"   ⚠️  Alias '{alias}' exists, using '{new_alias}'")

            return new_alias
        else:
            # Can't parse version - append "_v2"
            new_alias = f"{alias}_v2"
            if verbose:
                print(f"   ⚠️  Alias '{alias}' exists, using '{new_alias}'")
            return new_alias

    except Exception:
        # Model doesn't exist yet - use original alias
        return alias


def _generate_model_tags(model: "Model", custom_tags: Optional[dict] = None) -> dict:
    """
    Generate model tags from config and custom tags.

    Args:
        model: Model instance
        custom_tags: User-provided tags

    Returns:
        Merged tag dictionary
    """
    # Auto-generated tags
    tags = {
        "project": model.config.project.name,
        "model_type": model.model_type,
        "framework": _get_framework(model),
    }

    # Add experiment info if available
    if hasattr(model.config.experiment, "experiment_name"):
        tags["experiment"] = model.config.experiment.experiment_name

    if hasattr(model.config.experiment, "experiment_version"):
        tags["experiment_version"] = model.config.experiment.experiment_version

    # Add weights info
    if model.weights_name:
        tags["weights_name"] = model.weights_name

    # Merge with custom tags
    if custom_tags:
        tags.update(custom_tags)

    return tags


def _get_framework(model: "Model") -> str:
    """Get framework name based on model type"""
    if model.model_type == "pytorch":
        return "pytorch"
    elif model.model_type == "tensorflow":
        return "tensorflow"
    elif model.model_type == "keras":
        return "keras"
    else:
        return "python"


def _get_config_artifact_path(model: "Model") -> Optional[str]:
    """
    Get config.yaml path for artifact logging.

    Args:
        model: Model instance

    Returns:
        Path to config file, or None if not available
    """
    if (
        hasattr(model.config.experiment, "config_file_path")
        and model.config.experiment.config_file_path
    ):
        return str(model.config.experiment.config_file_path)
    return None


def _get_pyproject_artifact_path(model: "Model") -> Optional[str]:
    """
    Get pyproject.toml path for artifact logging.

    Args:
        model: Model instance

    Returns:
        Path to pyproject.toml file, or None if not available
    """
    if (
        hasattr(model.config.project, "pyproject_file_path")
        and model.config.project.pyproject_file_path
    ):
        return str(model.config.project.pyproject_file_path)
    return None


def _log_and_register_model(
    model: "Model",
    full_model_name: str,
    signature,
    input_example,
    tags: dict,
    verbose: bool = False,
    pyfunc_wrapper_class: Optional[type] = None,
):
    """
    Log model to MLflow and register to Unity Catalog.

    Args:
        model: Model instance
        full_model_name: Full UC model name
        signature: MLflow signature
        input_example: Input example
        tags: Model tags
        verbose: Print progress
        pyfunc_wrapper_class: Optional pyfunc wrapper for TF SavedModels

    Returns:
        mlflow.entities.model_registry.ModelVersion
    """
    import tempfile

    import mlflow

    with tempfile.TemporaryDirectory() as temp_dir:
        with mlflow.start_run() as run:
            model_path = "model"

            # Check if model is a YOLO model (not a raw torch.nn.Module)
            is_yolo = hasattr(model.model, "__class__") and "ultralytics" in str(type(model.model))

            # Get config artifact paths (will be bundled with model, not as run artifacts)
            config_path = _get_config_artifact_path(model)
            pyproject_path = _get_pyproject_artifact_path(model)

            if config_path and verbose:
                print(f"      Config artifact: {config_path}")
            if pyproject_path and verbose:
                print(f"      Pyproject artifact: {pyproject_path}")

            # Log model based on framework
            if model.model_type == "pytorch":
                from pathlib import Path

                import torch

                from bovi_core.ml.models.pytorch_model_wrapper import PyTorchModelWrapper

                framework = "Ultralytics YOLO" if is_yolo else "PyTorch"
                if verbose:
                    print(f"      Framework: {framework}")

                # Save PyTorch model to temp directory
                temp_model_path = Path(temp_dir) / "pytorch_model.pt"
                if is_yolo:
                    torch.save(model.model.model, temp_model_path)  # Inner nn.Module
                else:
                    torch.save(model.model, temp_model_path)

                # Build artifacts dict with model and config files
                # Config files are bundled WITH the model (not as run artifacts)
                # so they survive MLflow run deletion
                artifacts = {"model_path": str(temp_model_path)}
                if config_path:
                    artifacts["config_yaml"] = config_path
                if pyproject_path:
                    artifacts["pyproject_toml"] = pyproject_path

                mlflow.pyfunc.log_model(
                    python_model=PyTorchModelWrapper(),
                    name=model_path,
                    artifacts=artifacts,
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=[f"torch=={torch.__version__}", "numpy"],
                )

            elif model.model_type in ("tensorflow", "keras"):
                if verbose:
                    print("      Framework: TensorFlow/Keras")

                # Check if model.model is a loaded SavedModel or a Keras model
                is_saved_model = hasattr(model.model, "signatures")

                if is_saved_model and model.weights_location in [
                    "local",
                    "temp",
                    "workspace",
                ]:
                    # Use pyfunc wrapper for pure TensorFlow SavedModels
                    # MLflow tensorflow flavor only accepts Keras models, not SavedModels
                    from pathlib import Path

                    import tensorflow as tf

                    absolute_model_path = Path(model.weights_path).resolve()

                    # Require pyfunc_wrapper_class for SavedModels
                    if pyfunc_wrapper_class is None:
                        raise ValueError(
                            "pyfunc_wrapper_class is required for TensorFlow SavedModels. "
                            "Please provide a concrete TensorFlowSavedModelWrapper subclass "
                            "that implements get_input_name_mapping()."
                        )

                    # Build artifacts dict with model and config files
                    # Config files are bundled WITH the model (not as run artifacts)
                    # so they survive MLflow run deletion
                    artifacts = {"model_path": str(absolute_model_path)}
                    if config_path:
                        artifacts["config_yaml"] = config_path
                    if pyproject_path:
                        artifacts["pyproject_toml"] = pyproject_path

                    mlflow.pyfunc.log_model(
                        python_model=pyfunc_wrapper_class(),
                        name=model_path,
                        artifacts=artifacts,
                        signature=signature,
                        input_example=input_example,
                        pip_requirements=[f"tensorflow=={tf.__version__}", "numpy"],
                    )
                else:
                    # For Keras models, use pyfunc wrapper to bundle config files
                    from pathlib import Path

                    import tensorflow as tf

                    from bovi_core.ml.models.keras_model_wrapper import KerasModelWrapper

                    # Save Keras model to temp directory
                    keras_model_path = Path(temp_dir) / "keras_model"
                    model.model.save(keras_model_path)

                    # Build artifacts dict with model and config files
                    artifacts = {"model_path": str(keras_model_path)}
                    if config_path:
                        artifacts["config_yaml"] = config_path
                    if pyproject_path:
                        artifacts["pyproject_toml"] = pyproject_path

                    mlflow.pyfunc.log_model(
                        python_model=KerasModelWrapper(),
                        name=model_path,
                        artifacts=artifacts,
                        signature=signature,
                        input_example=input_example,
                        pip_requirements=[f"tensorflow=={tf.__version__}", "numpy"],
                    )

            else:
                raise NotImplementedError(
                    f"MLflow logging not yet implemented for model_type='{model.model_type}'"
                )

            # Register to Unity Catalog
            model_uri = f"runs:/{run.info.run_id}/{model_path}"

            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=full_model_name,
                tags=tags,
            )

            return model_version
