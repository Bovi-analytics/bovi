"""
Azure Blob Storage data source.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Union

from bovi_core.utils import blob_utils

from ..base import DataSource

if TYPE_CHECKING:
    from bovi_core.config import Config

logger = logging.getLogger(__name__)


class BlobImageSource(DataSource[bytes]):
    """
    Load images from Azure Blob Storage using existing blob_utils.

    Args:
        config: Config instance (container from config.project.blob_storage)
        prefix: Blob path prefix
        substring: Substring to filter blob names (uses list_blobs_by_pattern)
    """

    def __init__(
        self,
        config: "Config",
        prefix: str = "",
        substring: str = "",
    ):
        self.config = config
        self.prefix = prefix
        self.substring = substring

        # Use existing blob_utils to list blobs
        self.blob_paths = blob_utils.list_blobs_by_pattern(
            dir_path=prefix, substring=substring, config=config
        )
        logger.info(f"BlobImageSource: {len(self.blob_paths)} blobs at {prefix}")

    def __len__(self) -> int:
        return len(self.blob_paths)

    def load_item(self, key: Union[int, str]) -> bytes:
        """Download blob using blob_utils.get_file_blob"""
        blob_path = self.blob_paths[key] if isinstance(key, int) else key
        return blob_utils.get_file_blob(blob_path, config=self.config)

    def get_metadata(self, key: Union[int, str]) -> Dict[str, Any]:
        """Get blob metadata"""
        blob_path = self.blob_paths[key] if isinstance(key, int) else key

        # Extract label from path: prefix/label/image.jpg
        parts = blob_path.replace(self.prefix, "").strip("/").split("/")
        label = parts[0] if len(parts) > 1 else "unknown"

        return {
            "path": blob_path,
            "filename": parts[-1],
            "label": label,
            "index": self.blob_paths.index(blob_path) if blob_path in self.blob_paths else -1,
        }

    def get_keys(self) -> List[int]:
        return list(range(len(self.blob_paths)))
