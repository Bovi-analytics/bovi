"""YOLO model for cow detection inference.

Loads and manages ultralytics YOLO models with support for
multiple weight locations (local, blob, temp, workspace, Unity Catalog).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from bovi_core.ml import Model, ModelRegistry
from typing_extensions import override
from ultralytics import YOLO

if TYPE_CHECKING:
    from bovi_core.config import Config

    from bovi_yolo.predictors.yolo_predictor import YOLOPredictor

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the appropriate device for model inference.

    Returns:
        torch.device for CUDA if available, otherwise CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@ModelRegistry.register("yolo")
class YOLOModel(Model[YOLO]):
    """YOLO model for cow detection.

    Supports loading weights from multiple locations:
    - local: Direct filesystem path
    - temp: /local_disk0/tmp/ for distributed computing
    - workspace: Databricks workspace path
    - mlflow: MLflow registry with version/alias support
    - unity_catalog: Databricks Unity Catalog

    Example:
        >>> model = YOLOModel(
        ...     config=config,
        ...     weights_path="weights/yolo12n.pt",
        ...     predictor=predictor,
        ...     weights_location="local",
        ... )
        >>> model.initialize()
        >>> results = model(image)
    """

    model: YOLO

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run YOLO inference.

        Delegates to the loaded ultralytics YOLO model.

        Args:
            *args: Input data (images, paths, etc.).
            **kwargs: YOLO predict parameters (conf, iou, imgsz, etc.).

        Returns:
            ultralytics.Results list.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model(*args, **kwargs)

    device: torch.device

    @override
    def _set_model_types(self) -> None:
        """Set model type identifiers."""
        self.model_name = "yolo"
        self.device = get_device()

    @override
    def load_model(self) -> YOLO:
        """Load YOLO model from configured weights location.

        Returns:
            Loaded YOLO model instance.

        Raises:
            ValueError: If weights_location is unsupported.
            ImportError: If required dependencies are missing.
        """
        if self.weights_location in ("temp", "workspace", "local"):
            return self._load_from_path()
        elif self.weights_location == "mlflow":
            return self._load_from_mlflow()
        elif self.weights_location == "unity_catalog":
            return self._load_from_unity_catalog()
        else:
            raise ValueError(
                f"Unsupported weights_location: '{self.weights_location}'. "
                f"Use: local, temp, workspace, mlflow, unity_catalog."
            )

    def _load_from_path(self) -> YOLO:
        """Load YOLO model from a filesystem path.

        Returns:
            Loaded YOLO model.
        """
        self.model = YOLO(self.weights_path)
        logger.info(
            "Loaded YOLO from path: %s (type: %s)",
            self.weights_path,
            type(self.model).__name__,
        )
        return self.model

    def _load_from_mlflow(self) -> YOLO:
        """Load YOLO model from MLflow registry.

        Returns:
            YOLO model loaded from MLflow.

        Raises:
            ImportError: If ultralytics weight loader is unavailable.
        """
        try:
            from ultralytics.nn.tasks import (
                attempt_load_one_weight,  # pyright: ignore[reportAttributeAccessIssue]
            )
        except ImportError:
            raise ImportError(
                "MLflow loading requires attempt_load_one_weight "
                "which is not available in this ultralytics version."
            ) from None

        device = get_device()
        loaded_model, _ = attempt_load_one_weight(self.weights_path)
        loaded_model.float()
        loaded_model = loaded_model.to(device)
        loaded_model.eval()

        self.model = YOLO()
        self.model.model = loaded_model
        logger.info("Loaded YOLO from MLflow: %s", self.weights_path)
        return self.model

    def _load_from_unity_catalog(self) -> YOLO:
        """Load YOLO model from Unity Catalog using MLflow.

        Returns:
            YOLO model loaded from Unity Catalog.

        Raises:
            ValueError: If loading fails from both pytorch and pyfunc.
        """
        import mlflow

        mlflow.set_registry_uri("databricks-uc")
        device = get_device()

        try:
            loaded_model = mlflow.pytorch.load_model(self.weights_path)
            self.model = YOLO()
            self.model.model = loaded_model
            self.model.model = self.model.model.to(device)
            self.model.model.eval()
            logger.info("Loaded YOLO from Unity Catalog: %s", self.weights_path)
            return self.model

        except Exception as e:
            logger.warning("PyTorch load failed, trying pyfunc: %s", str(e))
            try:
                loaded_model = mlflow.pyfunc.load_model(self.weights_path)
                logger.info(
                    "Loaded as pyfunc from Unity Catalog: %s",
                    self.weights_path,
                )
                return loaded_model  # type: ignore[return-value]

            except Exception as e2:
                raise ValueError(
                    f"Failed to load from Unity Catalog: {e}, fallback error: {e2}"
                ) from e2

    @classmethod
    def load_from_unity_catalog(
        cls,
        model_uri: str | None = None,
        catalog: str = "projects",
        schema: str = "bovi_core",
        model_name: str = "yolo_cow_detection",
        alias: str = "Champion",
        download_weights: bool = False,
    ) -> YOLOModel:
        """Load YOLO model from Unity Catalog and reconstruct full Model + Predictor.

        This is the recommended way to load registered models. It:
        1. Downloads YOLO weights from UC (to MLflow cache, or local if download_weights=True)
        2. Downloads config.yaml artifact
        3. Reconstructs Config object
        4. Creates Predictor from config
        5. Creates Model with predictor
        6. Returns fully functional model with three-level returns

        Args:
            model_uri: Full UC URI (e.g., "models:/projects.bovi_core.model@Champion").
                If None, builds from catalog/schema/model_name/alias.
            catalog: UC catalog name (default: "projects").
            schema: UC schema name (default: "bovi_core").
            model_name: Model name (default: "yolo_cow_detection").
            alias: Model alias (default: "Champion").
            download_weights: If True, copy weights to local experiment directory.
                If False (default), weights stay in MLflow cache.

        Returns:
            YOLOModel instance with predictor ready.

        Example:
            >>> model = YOLOModel.load_from_unity_catalog(
            ...     model_uri="models:/projects.bovi_core.yolo_cow_detection@Champion"
            ... )
            >>> result = model.predict(image, return_format="rich")
        """
        import shutil
        from pathlib import Path

        import mlflow
        from bovi_core.config import Config
        from bovi_core.utils.path_utils import (
            get_experiment_paths,
            get_project_root,
        )

        mlflow.set_registry_uri("databricks-uc")

        # Build model URI if not provided
        if not model_uri:
            model_uri = f"models:/{catalog}.{schema}.{model_name}@{alias}"

        logger.info("Loading model from Unity Catalog: %s", model_uri)

        # Get model version details from UC
        client = mlflow.MlflowClient()
        model_name_full = model_uri.split("models:/")[1].split("@")[0]
        model_name_short = model_name_full.split(".")[-1]
        alias_name = model_uri.split("@")[1] if "@" in model_uri else None

        if alias_name:
            model_version = client.get_model_version_by_alias(model_name_full, alias_name)
        else:
            versions = client.search_model_versions(f"name='{model_name_full}'")
            model_version = versions[0]

        # Download model artifacts (includes config files bundled with model)
        logger.info("Downloading model artifacts from: %s", model_version.source)

        loaded_pyfunc = mlflow.pyfunc.load_model(model_uri)

        # Get artifacts from loaded model's wrapper
        python_model = loaded_pyfunc._model_impl.python_model
        artifacts = getattr(python_model, "_artifacts", {})

        # Config files are bundled as model artifacts
        config_yaml_artifact = Path(artifacts.get("config_yaml", ""))
        pyproject_artifact = Path(artifacts.get("pyproject_toml", ""))

        # Set up local experiment directory
        project_root = Path(get_project_root())
        exp_paths = get_experiment_paths(
            str(project_root),
            model_name_short,
            str(model_version.version),
        )
        version_dir = Path(exp_paths["dir"])
        config_dir = Path(exp_paths["config_dir"])
        weights_dir = Path(exp_paths["weights_dir"])

        config_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Local experiment dir: %s", version_dir)

        # Copy bundled config files to local experiment directory
        config_file = config_dir / "config.yaml"
        pyproject_file = config_dir / "pyproject.toml"

        if config_yaml_artifact.exists():
            shutil.copy(config_yaml_artifact, config_file)
        else:
            raise ValueError(
                "config.yaml not found in model artifacts. Re-register with bundled config."
            )

        if pyproject_artifact.exists():
            shutil.copy(pyproject_artifact, pyproject_file)
        else:
            raise ValueError(
                "pyproject.toml not found in model artifacts. Re-register with bundled config."
            )

        # Optionally copy weights locally for persistence
        local_weights_path = None
        if download_weights:
            model_artifact_path = Path(artifacts.get("model_path", ""))
            if model_artifact_path.exists():
                local_weights_path = weights_dir / "yolo_model.pt"
                shutil.copy(model_artifact_path, local_weights_path)
                logger.info("Weights saved to: %s", local_weights_path)
            else:
                logger.warning("Could not find model weights in artifacts to download")

        # Get the loaded YOLO model from pyfunc wrapper
        loaded_yolo_model = python_model.model

        # Reconstruct Config from downloaded files
        config = Config(
            config_file_path=str(config_file),
            project_file_path=str(pyproject_file),
        )

        # Create predictor from config
        from bovi_yolo.predictors.yolo_predictor import YOLOPredictor

        predictor = YOLOPredictor(config=config)

        # Use base class method to create instance from loaded model
        effective_weights_path = str(local_weights_path) if local_weights_path else model_uri
        model: YOLOModel = cls._create_from_loaded_model(  # pyright: ignore[reportAssignmentType]
            config=config,
            predictor=predictor,
            loaded_model=loaded_yolo_model,
            signature=None,
            weights_path=effective_weights_path,
            model_type="pytorch",
        )

        logger.info(
            "Model loaded from Unity Catalog: %s (version %s)",
            model_name_full,
            model_version.version,
        )

        return model

    @classmethod
    def from_config(
        cls,
        config: Config,
        predictor: YOLOPredictor | None = None,
        weights_path: str | None = None,
    ) -> YOLOModel:
        """Create YOLOModel from Config object.

        Args:
            config: Config instance with model settings.
            predictor: YOLOPredictor instance (required).
            weights_path: Optional override for weights path.

        Returns:
            Initialized YOLOModel instance.

        Raises:
            ValueError: If predictor is not provided or weights_path not found.
        """
        if predictor is None:
            raise ValueError("predictor is required. Please provide a YOLOPredictor instance.")

        model_cfg = config.experiment.models.yolo
        weights_location = model_cfg.default_weights_location

        if weights_path is None:
            weights_attr = f"{weights_location}_weights"
            weights_path = getattr(model_cfg, weights_attr).default
            if weights_path is None:
                raise ValueError(
                    f"No weights_path provided and none found in "
                    f"config at models.yolo.{weights_attr}.default"
                )

        model_type = getattr(model_cfg, "framework", "pytorch")

        return cls(
            config=config,
            weights_path=str(weights_path),
            predictor=predictor,
            model_type=model_type,
            weights_location=weights_location,
            verbose=getattr(config.experiment, "verbose", 0),
        )
