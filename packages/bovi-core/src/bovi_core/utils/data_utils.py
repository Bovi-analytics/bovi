"""
Data utilities for ML workflows.

Includes:
- Parquet conversion
- DataLoader benchmarking
- Dataset statistics

Note: Path resolution utilities have been moved to path_utils.py
"""

import logging
from typing import Any, Dict

# Re-export path utilities for backwards compatibility
from bovi_core.utils.path_utils import resolve_data_path

logger = logging.getLogger(__name__)

__all__ = [
    "resolve_data_path",
    "convert_blob_to_parquet",
    "benchmark_dataloader",
]


def convert_blob_to_parquet(
    config,
    blob_container: str,
    blob_prefix: str,
    output_table: str,
    file_pattern: str = "*.jpg",
    batch_size: int = 1000,
):
    """
    Convert blob images to Parquet in Unity Catalog.

    This is called automatically by dataloaders, but can be run
    manually for large datasets to prepare them ahead of time.

    Args:
        config: Config instance
        blob_container: Blob container name
        blob_prefix: Prefix/folder path in container
        output_table: Unity Catalog table (catalog.schema.table)
        file_pattern: Glob pattern for files
        batch_size: Number of images per batch

    Returns:
        Dict with conversion stats
    """
    # Placeholder - will implement in Phase 7
    logger.info(f"Converting {blob_container}/{blob_prefix} to {output_table}")
    logger.info("This is a placeholder - full implementation in Phase 7")
    return {"status": "not_implemented"}


def benchmark_dataloader(dataloader, num_batches: int = 100) -> Dict[str, Any]:
    """
    Measure dataloader throughput and identify bottlenecks.

    Args:
        dataloader: DataLoader to benchmark
        num_batches: Number of batches to process

    Returns:
        Dict with metrics:
        - samples_per_second
        - batches_per_second
        - avg_batch_time_ms
        - recommendation
    """
    # Placeholder - will implement in Phase 8
    logger.info(f"Benchmarking dataloader for {num_batches} batches")
    logger.info("This is a placeholder - full implementation in Phase 8")
    return {"status": "not_implemented"}
