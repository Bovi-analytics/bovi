from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict

from bovi_core.config import Config


class ModelConfig(BaseModel):
    """Immutable model-construction configuration for one model family."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    model_key: ClassVar[str]
    framework: str

    @classmethod
    def from_config(cls, config: Config) -> Self:
        model_node = getattr(config.experiment.models, cls.model_key)
        architecture = getattr(model_node, "architecture", {})
        values = {"framework": model_node.framework, **_public_values(architecture)}
        return cls.model_validate(values)


def _public_values(node: object) -> dict[str, object]:
    if isinstance(node, dict):
        return dict(node)
    values = getattr(node, "__dict__", {})
    return {key: value for key, value in values.items() if not key.startswith("_")}
