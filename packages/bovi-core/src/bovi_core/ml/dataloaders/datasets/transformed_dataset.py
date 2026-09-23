"""Apply explicit sample transforms after decoding and before batching."""

from collections.abc import Callable, Sequence
from typing import Any

from bovi_core.ml.dataloaders.datasets.base_dataset import Dataset


class TransformedDataset(Dataset):
    """Wrap a dataset without moving image decoding into the source.

    Record transforms belong on TransformedSource. Use this wrapper for
    transforms that need the decoded sample, such as image layout changes.
    Transforms execute lazily in order, including in native loader workers.
    """

    def __init__(
        self,
        dataset: Dataset,
        transforms: Sequence[Callable[[dict[str, Any]], dict[str, Any]]],
    ) -> None:
        super().__init__(dataset.source, dataset.config)
        self.dataset = dataset
        self.transforms = tuple(transforms)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    @property
    def metadata(self) -> dict[str, Any]:
        return self.dataset.metadata
