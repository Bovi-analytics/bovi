import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, List, Optional, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from bovi_core.config import Config
from bovi_core.utils.path_utils import resolve_data_path

if TYPE_CHECKING:
    from bovi_core.ml.dataloaders.base import Dataset
    from bovi_core.ml.predictors import PredictionInterface

logger = logging.getLogger(__name__)

# Define the base types that models can be
ModelType = TypeVar("ModelType", bound=Union[nn.Module, YOLO, Any])


class Model(Generic[ModelType], ABC):
    # Core configuration
    config: Config
    weights_name: Optional[str]
    model_type: str
    model_name: str
    weights_location: str  # Where the model weights are stored; can be local, blob, dbfs, mlflow
    config_location: Optional[str]  # Where the model config is stored; can be local, blob, or None
    verbose: int

    # Path attributes (set during initialization)
    weights_path: str
    weights_path_blob: str
    weights_path_dbfs: str
    weights_path_temp: str
    weights_filename: str
    temp_base: str

    # Config file attributes (optional, for models that need them)
    model_cfg_path: Optional[str]
    model_cfg_path_blob: str
    model_cfg_path_local: str
    model_cfg_path_dbfs: str
    model_cfg_path_temp: str
    config_filename: str

    # Model (device is framework-specific, defined in subclasses)
    model: ModelType

    # Prediction interface
    predictor: "PredictionInterface"

    def __init__(
        self,
        config: Config,
        weights_path: str,
        predictor: "PredictionInterface",
        model_cfg_path: Optional[str] = None,
        model_type: str = "pytorch",
        weights_name: Optional[str] = None,
        weights_location: str = "temp",
        verbose: int = 0,
    ):
        self.config = config
        self.weights_path = weights_path
        self.predictor = predictor
        self.model_cfg_path = model_cfg_path
        self.model_type = model_type
        self.weights_name = weights_name
        self.weights_location = weights_location
        self.verbose = verbose

        # Initialize blob client from config (optional for local-only usage)
        try:
            self.blob_container_client = config.container_client
        except (ValueError, AttributeError):
            self.blob_container_client = None
            if verbose:
                logger.info("Blob storage not configured - running in local-only mode")

        self.initialize()

    def initialize(self) -> None:
        self._set_model_types()
        try:
            self.load_model()
            # Give predictor access to the Model instance (not self.model)
            # This allows predictor to use the Model's __call__ method for consistent inference
            self.predictor.set_model_instance(self)
        except Exception as e:
            raise Exception(f"Failed to load model from path: {self.weights_path}. Error: {e}")

    @classmethod
    def _create_from_loaded_model(
        cls,
        config: Config,
        predictor: "PredictionInterface",
        loaded_model: Any,
        signature: Any,
        weights_path: str,
        model_type: str = "tensorflow",
        verbose: int = 1,
    ) -> "Model":
        """
        Create model instance from already-loaded model (e.g., from Unity Catalog).

        This bypasses normal __init__ and load_model() since we already have the model.
        Used by load_from_unity_catalog() implementations to avoid redundant loading.

        Args:
            config: Config object with model configuration
            predictor: Initialized predictor instance for this model
            loaded_model: Already-loaded model object (e.g., TensorFlow SavedModel)
            signature: Model signature for inference (e.g., TF serving signature)
            weights_path: Original path/URI where model was loaded from (for reference)
            model_type: Framework type ("tensorflow", "pytorch", etc.)
            verbose: Verbosity level

        Returns:
            Fully initialized Model instance ready for inference

        Example:
            >>> # In subclass load_from_unity_catalog:
            >>> predictor = MyPredictor(config=config)
            >>> return cls._create_from_loaded_model(
            ...     config=config,
            ...     predictor=predictor,
            ...     loaded_model=loaded_tf_model,
            ...     signature=loaded_tf_model.signatures["serving_default"],
            ...     weights_path=model_uri,
            ... )
        """
        instance = cls.__new__(cls)

        # Set core attributes (normally set by __init__)
        instance.config = config
        instance.predictor = predictor
        instance.weights_path = weights_path
        instance.weights_location = "unity_catalog"
        instance.model_type = model_type
        instance.model_cfg_path = None
        instance.weights_name = None
        instance.verbose = verbose
        try:
            instance.blob_container_client = config.container_client
        except (ValueError, AttributeError):
            instance.blob_container_client = None

        # Set model-specific types (device, model_name)
        instance._set_model_types()

        # Assign loaded model directly (skip load_model())
        instance.model = loaded_model
        instance._signature = signature

        # Connect predictor to model instance
        predictor.set_model_instance(instance)

        return instance

    def predict(self, data: Any, **kwargs: Any) -> Any:
        """
        Perform prediction using the model's predictor.

        This is a generic interface that delegates to the model-specific predictor.
        The input format depends on the model type:
        - Vision models: np.ndarray, List[np.ndarray], List[str] (blob paths)
        - Time-series models: Dict[str, Any] with structured inputs
        - Text models: str, List[str]

        Args:
            data: Input data in model-specific format
            **kwargs: Additional arguments to pass to the predictor, such as:
                - return_format: Output format ("raw", "base", "rich")
                - prompt: For SAM/Samurai models (dict mapping object IDs to bounding boxes)
                - Any other model-specific parameters

        Returns:
            Prediction result in format specified by the predictor
        """
        return self.predictor.predict(data, **kwargs)

    @abstractmethod
    def _set_model_types(self) -> None:
        pass

    @abstractmethod
    def load_model(self) -> ModelType:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Make model callable for inference.

        This provides a consistent interface across all frameworks:
        - TensorFlow: Calls the signature function
        - PyTorch: Calls the model's forward method
        - YOLO: Calls the YOLO predict method

        Subclasses must implement this to match their framework's calling convention.

        Args:
            *args: Positional arguments for the model
            **kwargs: Keyword arguments for the model

        Returns:
            Raw model output (framework-specific)

        Example:
            >>> # TensorFlow
            >>> output = model(input_1=x, input_2=y)
            >>>
            >>> # PyTorch
            >>> output = model(x, y)
        """
        pass

    # ========================================
    # Unity Catalog Registration Methods
    # ========================================

    def register_to_unity_catalog(
        self,
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
                If None, auto-generated as "/Users/{email}/{experiment_name}/{experiment_version}/run_{n}".

        Returns:
            mlflow.entities.model_registry.ModelVersion

        Example:
            >>> # Auto-generate everything from dataset
            >>> model.register_to_unity_catalog(
            ...     dataset=train_dataset,
            ...     alias="Champion",
            ...     description="YOLO model for cow detection"
            ... )

            >>> # Manual signature (advanced)
            >>> from mlflow.models import infer_signature
            >>> input_ex = {"image": np.zeros((1, 3, 640, 640))}
            >>> sig = infer_signature(input_ex)
            >>> model.register_to_unity_catalog(
            ...     input_example=input_ex,
            ...     signature=sig
            ... )

            >>> # Custom catalog and schema
            >>> model.register_to_unity_catalog(
            ...     catalog="production",
            ...     schema="models",
            ...     model_name="cow_detector_yolo",
            ...     dataset=dataset
            ... )

        Raises:
            ImportError: If mlflow is not installed
            ValueError: If neither dataset nor input_example is provided
        """
        from bovi_core.ml.models.unity_catalog import register_to_unity_catalog

        return register_to_unity_catalog(
            model=self,
            catalog=catalog,
            schema=schema,
            model_name=model_name,
            alias=alias,
            dataset=dataset,
            input_example=input_example,
            signature=signature,
            description=description,
            tags=tags,
            n_samples=n_samples,
            auto_increment_version=auto_increment_version,
            verbose=verbose,
            pyfunc_wrapper_class=pyfunc_wrapper_class,
            mlflow_experiment_name=mlflow_experiment_name,
        )