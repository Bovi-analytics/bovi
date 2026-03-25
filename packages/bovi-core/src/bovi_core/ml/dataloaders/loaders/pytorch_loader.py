"""
PyTorch DataLoader implementation.

Wraps torch.utils.data.DataLoader with sensible defaults.
Applies transforms via FrameworkAdapter collate function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from functools import partial
from typing import TYPE_CHECKING, Any

from ..adapters import FrameworkAdapter
from ..base import AbstractDataLoader, Dataset

if TYPE_CHECKING:
    from bovi_core.config import Config

logger = logging.getLogger(__name__)


class PyTorchDataLoader(AbstractDataLoader):
    """
    PyTorch DataLoader wrapper.

    Wraps torch.utils.data.DataLoader with optimized defaults for
    training and inference. Transforms are applied in the collate function.

    Key features:
    - Automatic prefetching with multiple workers
    - Pin memory for GPU training
    - Configurable batch size and shuffle
    - Transforms applied via FrameworkAdapter (HWC→CHW, uint8→float32)

    Args:
        dataset: Dataset to load from (returns raw NumPy)
        config: Config instance
        split: Dataset split ("train", "val", "test")
        model_name: Model name for config lookup
        transform: Optional Albumentations transform to apply per-sample
        batch_size: Batch size (overrides config)
        shuffle: Whether to shuffle (overrides config default)
        num_workers: Number of worker processes (overrides config)
        pin_memory: Pin memory for GPU (default: auto-detect)
        drop_last: Drop incomplete last batch (default: False)
        persistent_workers: Keep workers alive (default: True for training)
        prefetch_factor: Batches to prefetch per worker (default: 2)
        auto_transpose: Auto-convert HWC→CHW for images (default: True)
        auto_normalize: Auto-convert uint8→float32/255.0 (default: True)

    Example:
        ```python
        import albumentations as A
        from bovi_core.ml.dataloaders import (
            ImageDataset,
            LocalFileSource,
            PyTorchDataLoader,
        )

        # Create dataset (no transforms!)
        source = LocalFileSource("data/images", file_pattern="*.jpg")
        dataset = ImageDataset(source)  # Returns raw NumPy (H, W, C), uint8

        # Create transform
        transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Create loader WITH transform
        loader = PyTorchDataLoader(
            dataset,
            config=config,
            split="train",
            transform=transform,
            batch_size=32,
            num_workers=4,
        )

        # Iterate - images are automatically converted to (B, C, H, W) float32
        for batch in loader:
            images = batch["image"]  # (32, 3, 224, 224) float32 tensor
            labels = batch["label"]
            # ... training code
        ```
    """

    def __init__(
        self,
        dataset: Dataset,
        config: Config,
        split: str = "train",
        model_name: str | None = None,
        transform: Callable[..., Any] | None = None,
        batch_size: int | None = None,
        shuffle: bool | None = None,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        drop_last: bool = False,
        persistent_workers: bool | None = None,
        prefetch_factor: int = 2,
        auto_transpose: bool = True,
        auto_normalize: bool = True,
    ):
        super().__init__(dataset, config, split, model_name)

        self.transform = transform
        self.auto_transpose = auto_transpose
        self.auto_normalize = auto_normalize

        # Get config for this split (if available)
        split_config = None
        dataloader_config = None
        if model_name and hasattr(config.experiment, "models"):
            model_config = getattr(config.experiment.models, model_name, None)
            if model_config and hasattr(model_config, "dataloaders"):
                split_config = getattr(model_config.dataloaders, split, None)
                # Get nested dataloader config if it exists
                if split_config and hasattr(split_config, "dataloader"):
                    dataloader_config = split_config.dataloader

        # Determine parameters with fallback to config
        self.batch_size = batch_size or (
            dataloader_config.batch_size
            if dataloader_config and hasattr(dataloader_config, "batch_size")
            else split_config.batch_size
            if split_config and hasattr(split_config, "batch_size")
            else 32
        )

        # Default shuffle: True for train, False for val/test
        if shuffle is None:
            if dataloader_config and hasattr(dataloader_config, "shuffle"):
                shuffle = dataloader_config.shuffle
            elif split_config and hasattr(split_config, "shuffle"):
                shuffle = split_config.shuffle
            else:
                shuffle = split == "train"
        self.shuffle = shuffle

        # Default num_workers
        resolved_num_workers: int
        if num_workers is not None:
            resolved_num_workers = num_workers
        elif dataloader_config and hasattr(dataloader_config, "num_workers"):
            resolved_num_workers = int(dataloader_config.num_workers)
        elif split_config and hasattr(split_config, "num_workers"):
            resolved_num_workers = int(split_config.num_workers)
        else:
            resolved_num_workers = 4
        self.num_workers: int = resolved_num_workers

        # Auto-detect pin_memory (True if CUDA available)
        if pin_memory is None:
            try:
                import torch

                pin_memory = torch.cuda.is_available()
            except ImportError:
                pin_memory = False
        self.pin_memory = pin_memory

        # Persistent workers (keep alive between epochs)
        if persistent_workers is None:
            # Only use for training with workers
            persistent_workers = split == "train" and self.num_workers > 0
        self.persistent_workers = persistent_workers

        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor

        # Create PyTorch DataLoader
        self._pytorch_loader = None
        self._create_pytorch_loader()

        logger.info(
            f"PyTorchDataLoader ({split}): "
            f"batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, "
            f"num_workers={self.num_workers}, "
            f"transform={transform is not None}"
        )

    def _create_pytorch_loader(self) -> None:
        """Create the underlying PyTorch DataLoader"""
        try:
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError(
                "PyTorch is required for PyTorchDataLoader. Install with: pip install torch"
            )

        # Create collate function using FrameworkAdapter
        collate_fn = partial(
            FrameworkAdapter.numpy_to_pytorch_collate,
            transform=self.transform,
            auto_transpose=self.auto_transpose,
            auto_normalize=self.auto_normalize,
        )

        # Create DataLoader
        self._pytorch_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=collate_fn,
        )

    def get_pytorch_loader(self) -> Any:
        """
        Return PyTorch DataLoader instance.

        Returns:
            torch.utils.data.DataLoader
        """
        return self._pytorch_loader

    def get_tensorflow_dataset(self) -> None:
        """
        TensorFlow not supported by PyTorchDataLoader.

        Returns:
            None
        """
        return None

    def get_sklearn_iterator(self) -> None:
        """
        Simple iterator not needed (use PyTorch loader).

        Returns:
            None
        """
        return None

    def __iter__(self) -> Iterator[Any]:
        """Iterate over batches"""
        if self._pytorch_loader is None:
            raise RuntimeError("PyTorch DataLoader not initialized")
        return iter(self._pytorch_loader)

    def __len__(self) -> int:
        """Number of batches"""
        if self._pytorch_loader is None:
            raise RuntimeError("PyTorch DataLoader not initialized")
        return len(self._pytorch_loader)

    @property
    def num_batches(self) -> int:
        """Number of batches per epoch"""
        return len(self)

    @property
    def num_samples(self) -> int:
        """Total number of samples"""
        return len(self.dataset)
