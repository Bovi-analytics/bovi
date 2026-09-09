"""Native PyTorch batching; preprocessing is explicit on the dataset."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterator
from functools import partial
from typing import TYPE_CHECKING, Any

from bovi_core.ml.dataloaders.datasets.base_dataset import Dataset
from bovi_core.ml.dataloaders.loaders.base_loader import AbstractDataLoader

from ..batching import collate_pytorch_samples

if TYPE_CHECKING:
    from torch import Generator

    from bovi_core.config import Config

logger = logging.getLogger(__name__)


class PyTorchDataLoader(AbstractDataLoader):
    """Wrap torch.utils.data.DataLoader without image-specific defaults.

    Datasets return framework-neutral samples. Wrap a dataset in
    TransformedDataset to apply decoded-sample transforms before batching.
    The default collator preserves values, dtypes, layout and metadata; a
    supplied collate_fn replaces only batching, not dataset transforms.

    seed and generator are mutually exclusive. set_epoch pins sample order
    for repeated train/metrics passes, not augmentation randomness. Persistent
    workers retain transform RNG state. Worker initialization hooks must be
    picklable when using spawn.
    """

    def __init__(
        self,
        dataset: Dataset,
        config: Config,
        split: str = "train",
        model_name: str | None = None,
        batch_size: int | None = None,
        shuffle: bool | None = None,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        drop_last: bool = False,
        persistent_workers: bool | None = None,
        prefetch_factor: int = 2,
        preserve_keys: Collection[str] = ("metadata",),
        collate_fn: Callable[[list[Any]], Any] | None = None,
        seed: int | None = None,
        generator: Generator | None = None,
        worker_init_fn: Callable[[int], None] | None = None,
    ):
        super().__init__(dataset, config, split, model_name)

        self.preserve_keys = tuple(preserve_keys)
        self.collate_fn = collate_fn
        if seed is not None and generator is not None:
            raise ValueError("Pass seed or generator, not both")
        self.seed = seed
        self.generator = generator
        self.worker_init_fn = worker_init_fn
        self._epoch: int | None = None

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
            f"num_workers={self.num_workers}"
        )

    def _create_pytorch_loader(self) -> None:
        """Create the underlying PyTorch DataLoader"""
        try:
            from torch import Generator
            from torch.utils.data import DataLoader, RandomSampler
        except ImportError:
            raise ImportError(
                "PyTorch is required for PyTorchDataLoader. Install with: pip install torch"
            )

        if self.generator is None and self.seed is not None:
            self.generator = Generator().manual_seed(self.seed)
        if self.generator is not None:
            self.seed = self.generator.initial_seed()
        self._worker_generator = (
            Generator().manual_seed(self.seed) if self.seed is not None else None
        )
        # Separate sampling from worker startup RNG consumption, including with
        # persistent workers whose startup happens only on the first iteration.
        sampler = (
            RandomSampler(self.dataset, generator=self.generator)
            if self.shuffle and self.generator is not None
            else None
        )

        # Select batch assembly; sample transforms already belong to the dataset.
        collate_fn = (
            self.collate_fn
            if self.collate_fn is not None
            else partial(
                collate_pytorch_samples,
                preserve_keys=self.preserve_keys,
            )
        )

        # Create DataLoader
        self._pytorch_loader = DataLoader(
            dataset=self.dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=self.shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=collate_fn,
            generator=self._worker_generator,
            worker_init_fn=self.worker_init_fn,
        )

    def set_epoch(self, epoch: int) -> None:
        """Pin zero-based sample order; does not reset persistent-worker transforms."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if self.shuffle and self.seed is None:
            raise ValueError("set_epoch requires seed or generator when shuffling")
        self._epoch = epoch

    def __iter__(self) -> Iterator[Any]:
        """Iterate over batches"""
        if self._pytorch_loader is None:
            raise RuntimeError("PyTorch DataLoader not initialized")
        if self._epoch is not None and self.seed is not None:
            epoch_seed = (self.seed + self._epoch) % (2**64)
            if self.generator is not None:
                self.generator.manual_seed(epoch_seed)
            if self._worker_generator is not None:
                self._worker_generator.manual_seed(epoch_seed)
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
