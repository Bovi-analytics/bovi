"""YOLO image source factory.

Creates the appropriate DataSource based on configuration,
delegating to LocalFileSource or BlobImageSource from bovi-core.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from bovi_core.ml.dataloaders.sources import BlobImageSource, LocalFileSource

if TYPE_CHECKING:
    from bovi_core.config import Config
    from bovi_core.ml.dataloaders.base import DataSource

logger = logging.getLogger(__name__)


class YOLOImageSource:
    """Factory for creating YOLO image data sources.

    Delegates to LocalFileSource or BlobImageSource based on config.
    Provides static methods for direct construction.

    Example:
        >>> source = YOLOImageSource.from_config(config, split="train")
        >>> source = YOLOImageSource.from_local("data/images", "*.jpg")
    """

    @staticmethod
    def from_config(config: Config, split: str = "train") -> DataSource[bytes]:
        """Create source from experiment config.

        Reads source.type from dataloaders.{split}.source to determine
        whether to use LocalFileSource or BlobImageSource.

        Args:
            config: Config instance with experiment settings.
            split: Data split name ("train", "validation", "inference").

        Returns:
            DataSource instance for loading images.

        Raises:
            ValueError: If source type is unsupported.
        """
        dataloader_cfg = getattr(config.experiment.dataloaders, split)
        source_cfg = dataloader_cfg.source
        source_type = source_cfg.type

        if source_type == "local":
            root_dir = Path(config.project.project_root) / source_cfg.root_dir
            file_pattern = getattr(source_cfg, "file_pattern", "*.jp*g")
            logger.info(
                "Creating local source: %s (pattern: %s)",
                root_dir,
                file_pattern,
            )
            return LocalFileSource(
                root_dir=root_dir,
                file_pattern=file_pattern,
            )

        if source_type == "blob":
            prefix = getattr(source_cfg, "prefix", "")
            logger.info("Creating blob source with prefix: %s", prefix)
            return BlobImageSource(
                config=config,
                prefix=prefix,
            )

        raise ValueError(f"Unsupported source type: '{source_type}'. Use 'local' or 'blob'.")

    @staticmethod
    def from_local(
        root_dir: str | Path,
        file_pattern: str = "*.jp*g",
    ) -> LocalFileSource:
        """Create local file source for images.

        Args:
            root_dir: Root directory containing image files.
            file_pattern: Glob pattern to match image files.

        Returns:
            LocalFileSource instance.
        """
        logger.info(
            "Creating local source: %s (pattern: %s)",
            root_dir,
            file_pattern,
        )
        return LocalFileSource(
            root_dir=root_dir,
            file_pattern=file_pattern,
        )

    @staticmethod
    def from_blob(config: Config, prefix: str) -> BlobImageSource:
        """Create blob source for images from Azure Blob Storage.

        Args:
            config: Config instance with blob storage settings.
            prefix: Blob prefix path to filter images.

        Returns:
            BlobImageSource instance.
        """
        logger.info("Creating blob source with prefix: %s", prefix)
        return BlobImageSource(
            config=config,
            prefix=prefix,
        )
