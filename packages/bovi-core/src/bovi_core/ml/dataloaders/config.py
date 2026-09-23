"""Typed configuration contracts for model-specific dataloader factories."""

from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bovi_core.config import Config, ConfigNode, config_node_to_data


class LoaderSettings(BaseModel):
    """Framework-independent settings shared by batched dataloaders."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(default=32, gt=0)
    shuffle: bool = False
    seed: int | None = Field(default=None, ge=0, lt=2**32)


class DataLoaderConfig(BaseModel):
    """Immutable configuration for one model-specific data split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_key: ClassVar[str]
    split: str

    @field_validator("split")
    @classmethod
    def _validate_split(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("split must not be empty")
        return value

    @model_validator(mode="before")
    @classmethod
    def _validate_config_node(cls, value: object) -> object:
        return config_node_to_data(value) if isinstance(value, ConfigNode) else value

    @classmethod
    def from_config(cls, config: Config, split: str) -> Self:
        """Build one typed split config from the selected model YAML section."""
        model_key = getattr(cls, "model_key", None)
        if not model_key:
            raise TypeError(f"{cls.__name__} must define a model_key to use from_config()")

        model_node = getattr(config.experiment.models, model_key)
        split_node = getattr(model_node.dataloaders, split)
        values: dict[str, object] = {
            "split": split,
            **config_node_to_data(split_node),
        }
        if hasattr(model_node, "dataset"):
            values["dataset"] = config_node_to_data(model_node.dataset)

        return cls.model_validate(values)
