"""Strict selected data settings for the three tabular reference examples.

These schemas are opt-in at the example factory boundary, not a replacement
for legacy loader configuration or the Config singleton.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class JSONRecordsSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["json_records"]
    path: Path


class TabularDatasetSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_name: str = Field(min_length=1)


class TabularLoaderSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(default=32, gt=0)
    shuffle: bool | None = None
    seed: int = Field(default=42, ge=0, lt=2**32)


class TabularTransformSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class TabularSplitSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: JSONRecordsSettings
    transforms: list[TabularTransformSettings] = Field(default_factory=list)
    framework: Literal["sklearn", "pytorch", "tensorflow"] | None = None
    dataloader: TabularLoaderSettings = Field(default_factory=TabularLoaderSettings)


class TabularDataSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: TabularDatasetSettings
    split: TabularSplitSettings
