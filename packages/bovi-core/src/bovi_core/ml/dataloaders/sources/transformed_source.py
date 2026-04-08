"""Transformed source wrapper for lazy transform application."""

from typing import Any, Protocol


class DataSource(Protocol):
    """Protocol for data sources that can be wrapped with transforms."""

    def __len__(self) -> int:
        """Return the number of items in the source."""
        ...

    def load_item(self, index: int) -> dict[str, Any]:
        """Load a single item by index."""
        ...


class TransformedSource:
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

    def __init__(self, source: DataSource, transforms: list[Any]) -> None:
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

    def load_item(self, index: int) -> dict[str, Any]:
        """
        Load item and apply transforms.

        Args:
            index: Item index

        Returns:
            Transformed data dict

        """
        data = self.source.load_item(index)
        for transform in self.transforms:
            data = transform(data)
        return data

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to wrapped source."""
        return getattr(self.source, name)
