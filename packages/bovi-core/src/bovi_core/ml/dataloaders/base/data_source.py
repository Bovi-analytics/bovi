"""
Abstract base class for data sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Generic, TypeVar

# Type variable for the data returned by load_item
# - bytes for image sources
# - dict[str, object] for tabular/time-series sources
DataT = TypeVar("DataT")


class DataSource(ABC, Generic[DataT]):
    """
    Abstract interface for loading data from any source.

    Generic over DataT to support different data types:
    - DataSource[bytes] for image sources (raw bytes)
    - DataSource[dict[str, object]] for tabular/time-series sources

    Implementations define WHERE data comes from:
    - BlobImageSource: Azure Blob Storage
    - LocalFileSource: Local filesystem
    - UnityCatalogParquetSource: UC Parquet tables
    - SparkDataFrameSource: Spark DataFrames
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of items."""
        pass

    @abstractmethod
    def load_item(self, key: int | str) -> DataT:
        """
        Load raw data by index or identifier.

        Args:
            key: Index (int) or unique identifier (str)

        Returns:
            Data in the source's native format (bytes, dict, etc.)
        """
        pass

    @abstractmethod
    def get_metadata(self, key: int | str) -> dict[str, object]:
        """
        Get metadata without loading full data.

        Args:
            key: Index (int) or identifier (str)

        Returns:
            Metadata dict with keys like:
            - "path": Original path or identifier
            - "label": Classification label (if available)
            - "size": File size in bytes
            - Additional source-specific metadata
        """
        pass

    @abstractmethod
    def get_keys(self) -> list[int | str]:
        """
        Get list of all available keys.

        Returns:
            List of keys that can be passed to load_item()
        """
        pass

    def __iter__(self) -> Iterator[DataT]:
        """
        Iterate over all items in the data source.

        Default implementation iterates by index. Subclasses can override
        for more efficient iteration.
        """
        for i in range(len(self)):
            yield self.load_item(i)

    def close(self) -> None:
        """
        Clean up resources (optional override).

        Called when dataset is done using this source.
        """
        pass

    def __enter__(self) -> DataSource[DataT]:
        """Context manager support."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager cleanup."""
        self.close()
