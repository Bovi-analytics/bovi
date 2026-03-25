"""
Lactation Autoencoder Model implementation.

This module provides a TensorFlow/Keras autoencoder for milk production time-series prediction.
The model takes milk recordings, events, parity, and herd statistics as inputs.

The model architecture is already trained and saved as a TensorFlow SavedModel.
This class provides a wrapper to load and use the model within the bovi-core framework.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import tensorflow as tf
from bovi_core.config import Config
from bovi_core.ml import Model, ModelRegistry
from bovi_core.utils.path_utils import get_experiment_paths, get_project_root
from typing_extensions import override

if TYPE_CHECKING:
    from lactation_autoencoder.predictors.lactation_predictor import LactationPredictor


def get_device() -> str:
    """Get the appropriate device for model inference."""
    # TensorFlow automatically handles GPU if available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return "GPU"
    else:
        return "CPU"


@ModelRegistry.register("autoencoder")
class LactationAutoencoderModel(Model[tf.Module]):
    """
    Lactation Autoencoder model for milk production time-series prediction.

    This model implements an autoencoder architecture that processes:
    - Milk recordings (304-day sequence)
    - Event sequences (breeding, calving, etc.)
    - Parity information
    - Herd statistics (with hierarchical fallback)

    Example:
        >>> from bovi_core.config import Config
        >>> from lactation_autoencoder.models import LactationAutoencoderModel
        >>>
        >>> config = Config(experiment_name="lactation_autoencoder")
        >>> model = LactationAutoencoderModel(
        ...     config=config,
        ...     weights_path="path/to/weights",
        ...     predictor=predictor_instance
        ... )
    """

    device: str

    @override
    def _set_model_types(self) -> None:
        """Set model type identifiers."""
        self.model_name = "lactation_autoencoder"
        # TensorFlow handles device automatically, but we set a placeholder for base class
        self.device = "cpu"  # TensorFlow will use GPU if available

    @override
    def load_model(self) -> tf.Module:
        """
        Load the lactation autoencoder model from weights.

        Currently supports loading from local filesystem only.
        The model is loaded as a full TensorFlow SavedModel object.
        The signature is stored separately in _signature for inference.

        Returns:
            Full SavedModel object
        """
        if self.weights_location in ["temp", "workspace", "local"]:
            # Load full SavedModel
            try:
                saved_model = tf.saved_model.load(self.weights_path)
                self.model = saved_model  # Store full model
                self._signature = saved_model.signatures[
                    "serving_default"
                ]  # Signature for inference

                if self.verbose > 0:
                    print(f"✅ Loaded model from: {self.weights_path}")
                    print("   Model type: TensorFlow SavedModel")

                return self.model
            except Exception as e:
                raise ValueError(
                    f"Could not load model from {self.weights_path}. "
                    f"Expected TensorFlow SavedModel directory with saved_model.pb. Error: {e}"
                )
        else:
            raise ValueError(
                f"Unsupported weights_location: {self.weights_location}. "
                f"Currently only 'local', 'temp', and 'workspace' are supported."
            )

    @override
    def __call__(self, **kwargs: Any) -> Any:
        """
        Make model callable via TensorFlow signature function.

        This provides a consistent inference interface across all model types.
        For TensorFlow SavedModel, this calls the serving signature.

        Args:
            **kwargs: Keyword arguments matching the model's signature inputs
                     (e.g., input_11=..., input_12=..., input_13=..., input_15=...)

        Returns:
            Raw TensorFlow tensor output from the signature

        Example:
            >>> output = model(input_11=milk, input_12=parity, input_13=events, input_15=herd_stats)
        """
        if not hasattr(self, "_signature") or self._signature is None:
            raise RuntimeError(
                "Model signature not available. Ensure load_model() was called successfully."
            )
        return self._signature(**kwargs)

    @classmethod
    def load_from_unity_catalog(
        cls,
        model_uri: str | None = None,
        catalog: str = "projects",
        schema: str = "bovi_core",
        model_name: str = "lactation_autoencoder",
        alias: str = "Champion",
        download_weights: bool = False,
    ):
        """
        Load model from Unity Catalog and reconstruct full Model + Predictor.

        This is the recommended way to load registered models. It:
        1. Downloads TensorFlow weights from UC (to MLflow cache, or local if download_weights=True)
        2. Downloads config.yaml artifact
        3. Reconstructs Config object
        4. Creates Predictor from config
        5. Creates Model with predictor
        6. Returns fully functional model with three-level returns

        Args:
            model_uri: Full UC URI (e.g., "models:/projects.bovi_core.model@Champion").
                If None, builds from catalog/schema/model_name/alias.
            catalog: UC catalog name (default: "projects")
            schema: UC schema name (default: "bovi_core")
            model_name: Model name (default: "lactation_autoencoder")
            alias: Model alias (default: "Champion")
            download_weights: If True, copy weights to local experiment weights_dir for persistence.
                If False (default), weights stay in MLflow cache (faster, but not persistent).

        Returns:
            LactationAutoencoderModel instance with predictor ready

        Example:
            >>> # Load by URI (weights in MLflow cache)
            >>> model = LactationAutoencoderModel.load_from_unity_catalog(
            ...     model_uri="models:/projects.bovi_core.lactation_autoencoder@Champion"
            ... )
            >>>
            >>> # Load with persistent local weights
            >>> model = LactationAutoencoderModel.load_from_unity_catalog(
            ...     model_uri="models:/projects.bovi_core.lactation_autoencoder@Champion",
            ...     download_weights=True
            ... )
            >>>
            >>> # Use as normal with three-level returns
            >>> result = model.predict(data, return_format='rich')
            >>> result.plot_prediction()
        """
        # Set UC registry
        mlflow.set_registry_uri("databricks-uc")

        # Build model URI if not provided
        if not model_uri:
            model_uri = f"models:/{catalog}.{schema}.{model_name}@{alias}"

        print(f"Loading model from Unity Catalog: {model_uri}")

        # Get model version details from UC
        client = mlflow.MlflowClient()
        model_name_full = model_uri.split("models:/")[1].split("@")[0]
        # Extract short model name from full name
        # e.g., "projects.bovi_core.lactation_autoencoder" -> "lactation_autoencoder"
        model_name_short = model_name_full.split(".")[-1]
        alias_name = model_uri.split("@")[1] if "@" in model_uri else None

        if alias_name:
            model_version = client.get_model_version_by_alias(model_name_full, alias_name)
        else:
            versions = client.search_model_versions(f"name='{model_name_full}'")
            model_version = versions[0]

        # Download model artifacts (includes config files bundled with model)
        # This downloads from model.source, not run artifacts - survives run deletion
        print(f"   Downloading model artifacts from: {model_version.source}")

        # Load the pyfunc model first - this downloads all bundled artifacts
        loaded_pyfunc = mlflow.pyfunc.load_model(model_uri)

        # Get the artifacts from the loaded model's wrapper
        # The TensorFlowSavedModelWrapper stores artifacts in _artifacts during load_context
        python_model = loaded_pyfunc._model_impl.python_model
        artifacts = getattr(python_model, "_artifacts", {})

        # Config files are bundled as model artifacts (not run artifacts)
        config_yaml_artifact = Path(artifacts.get("config_yaml", ""))
        pyproject_artifact = Path(artifacts.get("pyproject_toml", ""))

        # Determine local experiment paths using get_experiment_paths

        project_root = Path(get_project_root())
        exp_paths = get_experiment_paths(
            str(project_root),
            model_name_short,
            str(model_version.version),
        )
        version_dir = Path(exp_paths["dir"])
        config_dir = Path(exp_paths["config_dir"])
        weights_dir = Path(exp_paths["weights_dir"])

        # Create directories
        config_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)

        print(f"   Local experiment dir: {version_dir}")

        # Copy bundled config files to local experiment config directory

        config_file = config_dir / "config.yaml"
        pyproject_file = config_dir / "pyproject.toml"

        if config_yaml_artifact.exists():
            shutil.copy(config_yaml_artifact, config_file)
        else:
            msg = "config.yaml not found in model artifacts. Re-register with bundled config."
            raise ValueError(msg)

        if pyproject_artifact.exists():
            shutil.copy(pyproject_artifact, pyproject_file)
        else:
            msg = "pyproject.toml not found in model artifacts. Re-register with bundled config."
            raise ValueError(msg)

        print(f"   Config: {config_file}")
        print(f"   Project: {pyproject_file}")

        # Optionally copy weights to local experiment directory for persistence
        local_weights_path = None
        if download_weights:
            model_artifact_path = Path(artifacts.get("model_path", ""))
            if model_artifact_path.exists():
                local_weights_path = weights_dir / "model"
                if local_weights_path.exists():
                    shutil.rmtree(local_weights_path)
                shutil.copytree(model_artifact_path, local_weights_path)
                print(f"   Weights: {local_weights_path}")
            else:
                print("   ⚠️  Could not find model weights in artifacts to download")

        loaded_tf_model = loaded_pyfunc._model_impl.python_model.model

        # Create Config object with downloaded artifact paths
        config = Config(config_file_path=str(config_file), project_file_path=str(pyproject_file))

        # Create predictor from config (local import to avoid circular dependency)
        from lactation_autoencoder.predictors.lactation_predictor import LactationPredictor

        predictor = LactationPredictor(config=config)

        # Use base class method to create instance from loaded model
        # If weights were downloaded locally, use that path; otherwise use the UC model URI
        effective_weights_path = str(local_weights_path) if local_weights_path else model_uri
        model = cls._create_from_loaded_model(
            config=config,
            predictor=predictor,
            loaded_model=loaded_tf_model,
            signature=loaded_tf_model.signatures["serving_default"],
            weights_path=effective_weights_path,
            model_type="tensorflow",
        )

        print("✅ Model loaded successfully from Unity Catalog")
        print(f"   Model: {model_name_full}")
        print(f"   Version: {model_version.version}")
        if alias_name:
            print(f"   Alias: {alias_name}")

        return model

    @classmethod
    def from_config(
        cls,
        config: Config,
        predictor: LactationPredictor | None = None,
        weights_path: str | None = None,
    ) -> LactationAutoencoderModel:
        """
        Create model instance from config.

        Args:
            config: Config object with model configuration
            predictor: LactationPredictor instance (required)
            weights_path: Optional override for weights path

        Returns:
            LactationAutoencoderModel instance

        Raises:
            ValueError: If predictor is not provided or weights_path not found

        Example:
            >>> from lactation_autoencoder import LactationPredictor
            >>> predictor = LactationPredictor(config=config)
            >>> model = LactationAutoencoderModel.from_config(
            ...     config=config,
            ...     predictor=predictor
            ... )
        """
        if predictor is None:
            raise ValueError("predictor is required. Please provide a LactationPredictor instance.")

        # Get model config from config.experiment.models.autoencoder
        model_config = config.experiment.models.autoencoder
        weights_location = getattr(model_config, "default_weights_location", "local")

        # Use provided weights_path or resolve from path templates
        if weights_path is None:
            weights_attr = f"{weights_location}_weights"
            weights_path = getattr(model_config, weights_attr).default
            if weights_path is None:
                raise ValueError(
                    f"No weights_path provided and none found in "
                    f"config at models.autoencoder.{weights_attr}.default"
                )

        model_type = getattr(model_config, "framework", "tensorflow")

        return cls(
            config=config,
            weights_path=weights_path,
            predictor=predictor,
            model_type=model_type,
            weights_location=weights_location,
            verbose=getattr(config.experiment, "verbose", 0),
        )
