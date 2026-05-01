"""Transformed source wrapper for lazy transform application."""

from __future__ import annotations

from typing import Any

from bovi_core.ml.dataloaders.base.data_source import DataSource


class TransformedSource(DataSource[dict[str, Any]]):
    """
    Wrapper that applies transforms when loading items.

    Follows the "Datasets are Dumb" principle by moving transform
    application to the source layer, before data reaches the dataset.

    Transforms are applied lazily (on each load_item call) rather than
    pre-computing all transformed data upfront.

    Example:
        >>> from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry
        >>>
        >>> source = MyDataSource(...)
        >>> transforms = TransformRegistry.from_config(
        ...     config.experiment.dataloaders.train.transforms
        ... )
        >>> transformed_source = TransformedSource(source, list(transforms.values()))
        >>> dataset = MyDataset(source=transformed_source, config=config)

    """

    def __init__(self, source: DataSource[dict[str, Any]], transforms: list[Any]) -> None:
        """
        Initialize transformed source.

        Args:
            source: Original data source implementing DataSource protocol
            transforms: List of transform instances to apply (in order)

        """
        self.source = source
        self.transforms = transforms

    def __len__(self) -> int:
        """Return number of items in source."""
        return len(self.source)

    def load_item(self, key: int | str) -> dict[str, Any]:
        """
        Load item and apply transforms.

        Args:
            key: Item index or identifier.

        Returns:
            Transformed data dict.

        """
        data = self.source.load_item(key)
        for transform in self.transforms:
            data = transform(data)
        return data

    def get_metadata(self, key: int | str) -> dict[str, object]:
        """Delegate metadata lookup to the wrapped source."""
        return self.source.get_metadata(key)

    def get_keys(self) -> list[int | str]:
        """Delegate key enumeration to the wrapped source."""
        return self.source.get_keys()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to wrapped source."""
        return getattr(self.source, name)
