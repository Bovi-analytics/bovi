"""
Abstract DataLoader interface.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from ..datasets.base_dataset import Dataset


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
        *,
        split: str = "train",
    ) -> None:
        self.dataset = dataset
        self.split = split

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """Iterate over batches."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Number of batches."""
        pass
