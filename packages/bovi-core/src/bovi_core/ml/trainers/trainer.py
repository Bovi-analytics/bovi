from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, TypeVar

from bovi_core.ml.dataloaders.base import AbstractDataLoader
from bovi_core.ml.models.model import Model

from .config import TrainingConfig
from .context import TrainingContext
from .results import TrainingResult

ConfigT = TypeVar("ConfigT", bound=TrainingConfig)
# A concrete Bovi Model subtype, not the wrapped native framework model.
ModelT = TypeVar("ModelT", bound=Model[Any])


class Trainer(ABC, Generic[ModelT, ConfigT]):
    def __init__(
        self,
        model: ModelT,
        dataloaders: Mapping[str, AbstractDataLoader],
        config: ConfigT,
        context: TrainingContext | None = None,
    ) -> None:
        # TODO: Add lifecycle hooks once the concrete trainer contract is settled.
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.context = context

    @abstractmethod
    def train(self) -> TrainingResult:
        """Train the model and return its result."""
        raise NotImplementedError
