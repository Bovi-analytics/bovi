from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, model_validator

from bovi_core.config import Config, ConfigNode, config_node_to_data


class ModelConfig(BaseModel):
    """Immutable model-construction configuration for one model family."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_key: ClassVar[str]
    framework: str

    @model_validator(mode="before")
    @classmethod
    def _validate_config_node(cls, value: object) -> object:
        return config_node_to_data(value) if isinstance(value, ConfigNode) else value

    @classmethod
    def from_config(cls, config: Config) -> Self:
        model_node = getattr(config.experiment.models, cls.model_key)
        architecture = getattr(model_node, "architecture", {})
        values = {"framework": model_node.framework, **config_node_to_data(architecture)}
        return cls.model_validate(values)
