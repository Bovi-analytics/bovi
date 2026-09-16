"""YOLO source construction from typed source settings."""

from __future__ import annotations

from bovi_core.ml.dataloaders.sources import LocalFileSource
from bovi_core.ml.dataloaders.sources.base_source import DataSource

from .config import YOLOSourceSettings


def create_source(source_config: YOLOSourceSettings) -> DataSource[bytes]:
    """Create a local image source from validated settings."""
    return LocalFileSource(
        root_dir=source_config.root_dir,
        file_pattern=source_config.file_pattern,
        recursive=source_config.recursive,
    )
