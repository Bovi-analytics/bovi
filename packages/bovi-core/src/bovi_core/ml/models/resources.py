from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Mapping, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

PayloadT = TypeVar("PayloadT")


class ModelArtifactReference(BaseModel):
    """Portable reference to a model artifact in external storage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str = Field(min_length=1)
    format: str = Field(min_length=1)
    checksum: str | None = Field(default=None, min_length=1)


class CheckpointReference(BaseModel):
    """Portable checkpoint reference; recovery guarantees are format-specific."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str = Field(min_length=1)
    format: str = Field(min_length=1)
    checksum: str | None = Field(default=None, min_length=1)


@dataclass(frozen=True, slots=True)
class ResolvedCheckpoint(Generic[PayloadT]):
    """Checkpoint content resolved from storage for framework-specific restoration."""

    format: str
    source_uri: str
    local_path: Path | None = None
    payload: PayloadT | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.local_path is None and self.payload is None:
            raise ValueError("a resolved checkpoint requires a local_path or payload")


@dataclass(frozen=True, slots=True)
class ResolvedModelArtifact(Generic[PayloadT]):
    """Deployment artifact resolved from storage for framework-specific loading."""

    format: str
    source_uri: str
    local_path: Path | None = None
    payload: PayloadT | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.local_path is None and self.payload is None:
            raise ValueError("a resolved model artifact requires a local_path or payload")


class ModelArtifactResolver(Protocol):
    """Materialize an external model artifact without understanding its framework."""

    def resolve(self, reference: ModelArtifactReference) -> ResolvedModelArtifact[object]: ...


class CheckpointResolver(Protocol):
    """Materialize checkpoint data without interpreting framework state."""

    def resolve(self, reference: CheckpointReference) -> ResolvedCheckpoint[object]: ...
