"""
Local filesystem data source.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

from ..base import DataSource

logger = logging.getLogger(__name__)


class LocalFileSource(DataSource[bytes]):
    """
    Load data from local filesystem.

    Args:
        root_dir: Root directory containing files
        file_pattern: Glob pattern (e.g., "*.jpg", "**/*.png")
        recursive: Whether to search recursively
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        file_pattern: str = "*",
        recursive: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.file_pattern = file_pattern
        self.recursive = recursive

        # Find all files matching pattern
        self.file_paths = self._find_files()
        logger.info(f"Found {len(self.file_paths)} files in {root_dir}")

    def _find_files(self) -> List[Path]:
        """Find all files matching pattern"""
        if self.recursive:
            pattern = f"**/{self.file_pattern}"
        else:
            pattern = self.file_pattern

        return sorted(self.root_dir.glob(pattern))

    def __len__(self) -> int:
        return len(self.file_paths)

    def load_item(self, key: Union[int, str]) -> bytes:
        """Load file as bytes"""
        if isinstance(key, int):
            file_path = self.file_paths[key]
        else:
            file_path = Path(key)

        return file_path.read_bytes()

    def get_metadata(self, key: Union[int, str]) -> Dict[str, Any]:
        """Get file metadata"""
        if isinstance(key, int):
            file_path = self.file_paths[key]
        else:
            file_path = Path(key)

        stat = file_path.stat()

        # Extract label from parent directory name
        label = file_path.parent.name

        return {
            "path": str(file_path),
            "filename": file_path.name,
            "label": label,
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }

    def get_keys(self) -> List[int]:
        return list(range(len(self.file_paths)))
