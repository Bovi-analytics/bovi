from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar, Union

import numpy as np
import torch
from azure.storage.blob import ContainerClient

from bovi_core.config import Config

if TYPE_CHECKING:
    from ml.models.model import Model

# TypeVar for the return type of predict method
T = TypeVar("T")


class Predictor(ABC):
    config: Config
    blob_container_client: ContainerClient
    device: torch.device
    model: "Model"
    model_type: str
    location: str
    weights_path: Path
    save_weights: Union[str, bool]

    def __init__(
        self,
        config: Config,
        model,
        location: str = "blob",
        save_results: bool = False,
        save_location: str = "blob",
        save_path: Optional[str] = None,
        verbose: int = 0,
    ):
        self.config = config
        self.blob_container_client = config.container_client
        self.model = model
        self.location = location
        self.save_results = save_results
        self.save_location = save_location
        self.save_path = save_path
        self.verbose = verbose
        self.initialize()

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def predict(
        self,
        data: Union[np.ndarray, List[np.ndarray], List[str]],
        prompt: Optional[Dict[str, Any]] = None,
    ) -> T:
        """
        Perform prediction on the provided data.

        Args:
            data: Input data in one of these formats:
                - Single image as numpy array
                - List of images as numpy arrays
                - List of blob paths as strings
            prompt: Optional dictionary mapping object IDs to prompts (e.g., bounding boxes)
                    Only required for models that use prompts (SAM, Samurai)

        Returns:
            Prediction result(s) specific to the model type
        """
        pass
