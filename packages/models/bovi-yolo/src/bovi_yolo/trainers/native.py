"""Conversion helpers at the Bovi/Ultralytics result boundary."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bovi_core.ml import EpochResult
from ultralytics.utils import YAML

METRIC_NAMES = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50-95(B)": "map50_95",
}


def scalar_metrics(value: object) -> dict[str, float]:
    """Extract finite scalar metrics from an Ultralytics metric object."""
    raw = getattr(value, "results_dict", value)
    if not isinstance(raw, Mapping):
        return {}

    metrics: dict[str, float] = {}
    for key, item in raw.items():
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            metrics[METRIC_NAMES.get(str(key).strip(), str(key).strip())] = number
    return metrics


def read_epoch_results(csv_path: Path, fallback: object) -> tuple[EpochResult, ...]:
    """Read complete native epoch history, falling back to final metrics."""
    rows: list[EpochResult] = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for index, row in enumerate(csv.DictReader(stream), start=1):
                metrics = scalar_metrics(
                    {key: value for key, value in row.items() if key and key.strip() != "epoch"}
                )
                if metrics:
                    rows.append(EpochResult(epoch=index, metrics=metrics))
    if rows:
        return tuple(rows)

    metrics = scalar_metrics(fallback)
    return (EpochResult(epoch=1, metrics=metrics),) if metrics else ()


def best_epoch(history: tuple[EpochResult, ...]) -> int | None:
    """Select the epoch using Ultralytics' detection fitness weighting."""
    candidates = [
        (
            0.1 * epoch.metrics.get("map50", 0.0) + 0.9 * epoch.metrics.get("map50_95", 0.0),
            epoch.epoch,
        )
        for epoch in history
        if "map50" in epoch.metrics or "map50_95" in epoch.metrics
    ]
    if candidates:
        return max(candidates)[1]
    return history[-1].epoch if history else None


def dataset_size(native_owner: object, loader_name: str) -> int | None:
    """Read a native loader's dataset size without depending on its concrete type."""
    loader = getattr(native_owner, loader_name, None)
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    try:
        return len(dataset)  # type: ignore[arg-type]
    except TypeError:
        return None


def native_save_dir(native_model: Any) -> Path:
    trainer = getattr(native_model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("Ultralytics did not expose its training output directory")
    return Path(save_dir)


def count_split_images(dataset_yaml_path: Path, split: str) -> int:
    """Count local images declared by an Ultralytics dataset YAML without downloads."""
    if not dataset_yaml_path.is_file():
        return 0
    data = YAML.load(str(dataset_yaml_path))
    raw_root = data.get("path")
    root = Path(raw_root) if raw_root else dataset_yaml_path.parent
    if not root.is_absolute():
        root = (dataset_yaml_path.parent / root).resolve()
    sources = data.get(split) or ()
    if isinstance(sources, str):
        sources = (sources,)

    suffixes = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
    count = 0
    for source in sources:
        path = Path(source)
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            count += sum(
                candidate.is_file() and candidate.suffix.lower() in suffixes
                for candidate in path.rglob("*")
            )
        elif path.suffix.lower() == ".txt" and path.is_file():
            count += sum(
                bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
            )
        elif path.is_file() and path.suffix.lower() in suffixes:
            count += 1
    return count
