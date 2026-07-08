"""
Abstract DataLoader interface.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from bovi_core.config import Config

    from .dataset import Dataset


class AbstractDataLoader(ABC):
    """
    Abstract dataloader interface.

    Concrete loaders may wrap framework-native loaders internally, but the
    shared contract is intentionally small: callers can iterate batches and ask
    for the number of batches.
    """

    def __init__(
        self,
        dataset: "Dataset",
        config: "Config",
        split: str = "train",
        model_name: Optional[str] = None,
    ):
        self.dataset = dataset
        self.config = config
        self.split = split
        self.model_name = model_name

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """Iterate over batches."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Number of batches."""
        pass
