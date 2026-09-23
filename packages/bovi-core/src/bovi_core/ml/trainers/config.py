from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, model_validator

from bovi_core.config import Config, ConfigNode, config_node_to_data


class TrainingConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    # The key that is used to get the right model metadata from the experiment YAML
    # Classvar as each concrete config instance will read from the same key/modeltype
    model_key: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def _validate_config_node(cls, value: object) -> object:
        return config_node_to_data(value) if isinstance(value, ConfigNode) else value

    @classmethod
    def from_config(cls, config: Config) -> Self:
        if cls.model_key is None:
            raise TypeError(f"{cls.__name__} must define a model_key to use from_config()")
        model_node = getattr(config.experiment.models, cls.model_key)
        return cls.model_validate(
            config_node_to_data(model_node.training),
        )


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    model_key: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def _validate_config_node(cls, value: object) -> object:
        return config_node_to_data(value) if isinstance(value, ConfigNode) else value

    @classmethod
    def from_config(cls, config: Config) -> Self:
        if cls.model_key is None:
            raise TypeError(f"{cls.__name__} must define a model_key to use from_config()")
        model_node = getattr(config.experiment.models, cls.model_key)
        return cls.model_validate(
            config_node_to_data(model_node.evaluation),
        )
