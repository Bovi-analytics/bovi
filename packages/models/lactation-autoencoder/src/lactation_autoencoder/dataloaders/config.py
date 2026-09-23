"""Typed configuration for one lactation-autoencoder dataloader split."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from bovi_core.config import Config
from bovi_core.ml.dataloaders import DataLoaderConfig, LoaderSettings
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LactationDatasetSettings(BaseModel):
    """Model-wide lactation dataset settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_days: int = Field(default=304, gt=0)
    keep_in_memory: bool = True


class LactationJSONSourceSettings(BaseModel):
    """JSON record source settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["lactation_json"]
    json_root_dir: Path
    file_pattern: str = "*.json"


class LactationTransformSettings(BaseModel):
    """One transform specification consumed by the transform registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class LactationLoaderSettings(LoaderSettings):
    """NumPy batching settings used by the autoencoder pipeline."""

    seed: int = Field(default=42, ge=0, lt=2**32)


class LactationAutoencoderDataLoaderConfig(DataLoaderConfig):
    """Validated configuration for exactly one autoencoder data split."""

    model_key: ClassVar[str] = "autoencoder"

    dataset: LactationDatasetSettings
    source: LactationJSONSourceSettings
    transforms: tuple[LactationTransformSettings, ...] = ()
    dataloader: LactationLoaderSettings = Field(default_factory=LactationLoaderSettings)

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
        json_root_dir = result.source.json_root_dir
        if not json_root_dir.is_absolute():
            json_root_dir = Path(config.project.project_root) / json_root_dir
        return result.model_copy(
            update={"source": result.source.model_copy(update={"json_root_dir": json_root_dir})}
        )
