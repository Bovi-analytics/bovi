"""Typed configuration for one scikit SGD data split."""

from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from bovi_core.config import Config
from bovi_core.ml.dataloaders import DataLoaderConfig, LoaderSettings
from pydantic import BaseModel, ConfigDict, Field


class JSONRecordsSourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["json_records"]
    path: Path


class TabularDatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_name: str = Field(min_length=1)


class TransformConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class ScikitLoaderSettings(LoaderSettings):
    seed: int = Field(default=42, ge=0, lt=2**32)


class ScikitSGDDataLoaderConfig(DataLoaderConfig):
    """Complete, immutable configuration for one data split."""

    model_key: ClassVar[str] = "scikit_sgd"

    dataset: TabularDatasetConfig
    source: JSONRecordsSourceConfig
    transforms: tuple[TransformConfig, ...] = ()
    dataloader: ScikitLoaderSettings = Field(default_factory=ScikitLoaderSettings)

    @classmethod
    def from_config(cls, config: Config, split: str) -> Self:
        result = super().from_config(config, split)
        source_path = result.source.path
        if not source_path.is_absolute():
            source_path = Path(config.project.project_root) / source_path
        return result.model_copy(
            update={"source": result.source.model_copy(update={"path": source_path})}
        )
