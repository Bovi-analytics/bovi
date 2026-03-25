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
    Abstract dataloader wrapping framework-native loaders.

    Key principle: Don't reinvent optimized libraries!
    Wrap torch.utils.data.DataLoader, tf.data.Dataset, etc.
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
    def get_pytorch_loader(self) -> Optional[Any]:
        """
        Return PyTorch DataLoader (or None if not supported).

        Returns:
            torch.utils.data.DataLoader instance
        """
        pass

    @abstractmethod
    def get_tensorflow_dataset(self) -> Optional[Any]:
        """
        Return TensorFlow Dataset (or None if not supported).

        Returns:
            tf.data.Dataset instance
        """
        pass

    @abstractmethod
    def get_sklearn_iterator(self) -> Optional[Iterator]:
        """
        Return simple iterator (or None if not supported).

        For sklearn and simple use cases.
        """
        pass

    def __iter__(self):
        """Default iteration using primary framework"""
        loader = self.get_pytorch_loader()
        if loader is not None:
            return iter(loader)

        loader = self.get_tensorflow_dataset()
        if loader is not None:
            return iter(loader)

        loader = self.get_sklearn_iterator()
        if loader is not None:
            return loader

        raise NotImplementedError("No framework loader available")

    def __len__(self):
        """Number of batches"""
        try:
            # Try to get model-specific dataloader config
            if self.model_name and hasattr(self.config.experiment, "models"):
                model_config = getattr(self.config.experiment.models, self.model_name, None)
                if model_config and hasattr(model_config, "dataloaders"):
                    split_config = getattr(model_config.dataloaders, self.split, None)
                    if split_config and hasattr(split_config, "dataloader"):
                        batch_size = split_config.dataloader.batch_size
                        return len(self.dataset) // batch_size
            # Fallback: return dataset length
            return len(self.dataset)
        except (AttributeError, ZeroDivisionError):
            # If config not available or batch_size is 0, return dataset length
            return len(self.dataset)
