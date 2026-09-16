"""Typed configuration for one YOLO dataloader split."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from bovi_core.config import Config
from bovi_core.ml.dataloaders import DataLoaderConfig, LoaderSettings
from pydantic import BaseModel, ConfigDict, Field, model_validator


class YOLODatasetSettings(BaseModel):
    """Model-wide dataset behavior shared by all YOLO splits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    return_metadata: Literal[True] = True


class YOLOLocalSourceSettings(BaseModel):
    """Local image source settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["local"]
    root_dir: Path
    file_pattern: str = "*.jp*g"
    recursive: bool = True


YOLOSourceSettings = YOLOLocalSourceSettings


class YOLOTransformSettings(BaseModel):
    """One transform specification consumed by the transform registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class YOLOLoaderSettings(LoaderSettings):
    """PyTorch-specific loader settings for YOLO."""

    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool | None = None
    drop_last: bool = False
    persistent_workers: bool | None = None
    prefetch_factor: int = Field(default=2, gt=0)


class YOLODataLoaderConfig(DataLoaderConfig):
    """Validated configuration for exactly one YOLO data split."""

    model_key: ClassVar[str] = "yolo"

    dataset: YOLODatasetSettings
    source: YOLOSourceSettings
    transforms: tuple[YOLOTransformSettings, ...] = ()
    transform_backend: Literal["albumentations"] = "albumentations"
    dataloader: YOLOLoaderSettings = Field(default_factory=YOLOLoaderSettings)

    @model_validator(mode="before")
    @classmethod
    def _apply_split_defaults(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        values = dict(value)
        loader = dict(values.get("dataloader") or {})
        loader.setdefault("shuffle", values.get("split") == "train")
        values["dataloader"] = loader
        return values

    @classmethod
    def from_config(cls, config: Config, split: str) -> Self:
        result = super().from_config(config, split)
        source = result.source
        root_dir = source.root_dir
        if not root_dir.is_absolute():
            root_dir = Path(config.project.project_root) / root_dir
        source = source.model_copy(update={"root_dir": root_dir})
        return result.model_copy(update={"source": source})
