from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .config import ModelConfig

NativeModelT = TypeVar("NativeModelT")
ModelConfigT = TypeVar("ModelConfigT", bound=ModelConfig)


class Model(ABC, Generic[NativeModelT, ModelConfigT]):
    """Framework-neutral runtime wrapper around an instantiated native model.

    Model construction, checkpoint restoration, artifact loading, prediction,
    and export are deliberately owned by separate collaborators.
    """

    def __init__(self, native_model: NativeModelT, config: ModelConfigT) -> None:
        self.native_model = native_model
        self.config = config

    @abstractmethod
    def __call__(self, *args: object, **kwargs: object) -> object:
        """Invoke the native model using its framework-specific call convention."""
        raise NotImplementedError
