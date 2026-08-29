from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict

from bovi_core.config import Config


class TrainingConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # The key that is used to get the right model metadata from the experiment YAML
    # Classvar as each concrete config instance will read from the same key/modeltype
    model_key: ClassVar[str]

    @classmethod
    def from_config(cls, config: Config) -> Self:
        if cls.model_key is None:
            raise TypeError(f"{cls.__name__} must define a model_key to use from_config()")
        model_node = getattr(config.experiment.models, cls.model_key)
        return cls.model_validate(
            model_node.training,
        )
