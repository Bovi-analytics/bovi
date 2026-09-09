"""YOLO source construction from experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bovi_core.ml.dataloaders.sources import BlobImageSource, LocalFileSource

if TYPE_CHECKING:
    from bovi_core.config import Config
    from bovi_core.ml.dataloaders.base import DataSource


def create_source(config: Config, split: str) -> DataSource[bytes]:
    """Create the configured local or blob image source for one split."""
    source_config = getattr(config.experiment.models.yolo.dataloaders, split).source

    if source_config.type == "local":
        root_dir = Path(source_config.root_dir)
        if not root_dir.is_absolute():
            root_dir = Path(config.project.project_root) / root_dir
        return LocalFileSource(
            root_dir=root_dir,
            file_pattern=getattr(source_config, "file_pattern", "*.jp*g"),
            recursive=bool(getattr(source_config, "recursive", True)),
        )

    if source_config.type == "blob":
        return BlobImageSource(
            config=config,
            prefix=getattr(source_config, "prefix", ""),
            substring=getattr(source_config, "substring", ""),
        )

    raise ValueError(
        f"Unsupported YOLO source type: {source_config.type!r}. Expected 'local' or 'blob'."
    )
