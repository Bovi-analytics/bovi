"""Atomic local checkpoint bundles; native serialization stays with the caller.

Checksums detect damage, not trustworthiness. Only load trusted native payloads.
Published version directories are never reused or updated by this module.
"""

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote, urlparse
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .resources import CheckpointReference, ResolvedCheckpoint


class CheckpointManifest(BaseModel):
    """All bundle payloads, their integrity, and the supported recovery scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    format: str = Field(min_length=1)
    entrypoint: str
    resume_scope: Literal["weights_only"] = "weights_only"
    files: dict[str, str] = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _relative_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Invalid checkpoint file path: {name!r}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"Checkpoint file escapes bundle: {name!r}")
    return path


class LocalCheckpointStore:
    """Write a complete bundle on one filesystem, then atomically publish it.

    The writer receives a private directory. It must finish and close every file
    before returning. A failed write cannot modify any published checkpoint.
    POSIX directory entries and storage ancestors are synced. Other platforms
    guarantee atomic rename, not power-loss durability. A post-rename sync error
    fails the save and leaves an unreferenced bundle; no reference is returned
    with uncertain durability.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()

    def save(
        self,
        name: str,
        format: str,
        write: Callable[[Path], object],
        *,
        entrypoint: str,
        metadata: dict[str, object] | None = None,
    ) -> CheckpointReference:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Checkpoint name must be a single path component")
        self.directory.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.directory))
        published = self.directory / f"{name}-{uuid4().hex}"
        try:
            _relative_file(staging, entrypoint)
            write(staging)
            files = {}
            for path in staging.rglob("*"):
                if path.is_symlink():
                    raise ValueError("Checkpoint bundles cannot contain symlinks")
                if path.is_file():
                    key = path.relative_to(staging).as_posix()
                    if key == "manifest.json":
                        raise ValueError("manifest.json is reserved for the checkpoint store")
                    _relative_file(staging, key)
                    files[key] = _digest(path)
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
            if entrypoint not in files:
                raise ValueError("Checkpoint writer did not create the entrypoint")
            manifest = CheckpointManifest(
                format=format, entrypoint=entrypoint, files=files, metadata=metadata or {}
            )
            manifest_path = staging / "manifest.json"
            with manifest_path.open("w", encoding="utf-8") as stream:
                stream.write(manifest.model_dump_json(indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            checksum = _digest(manifest_path)
            for directory in sorted(
                (path for path in staging.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                _sync_directory(directory)
            _sync_directory(staging)
            staging.rename(published)
            # Sync the whole chain, including directories created by a caller or
            # left behind by an earlier failed attempt whose sync never finished.
            for directory in (self.directory, *self.directory.parents):
                _sync_directory(directory)
            return CheckpointReference(uri=published.as_uri(), format=format, checksum=checksum)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


class LocalCheckpointResolver:
    """Verify a local bundle before handing its native entrypoint to a provider."""

    def resolve(self, reference: CheckpointReference) -> ResolvedCheckpoint[object]:
        uri = urlparse(reference.uri)
        if uri.scheme not in {"", "file"} or uri.netloc not in {"", "localhost"}:
            raise ValueError("Expected a local checkpoint URI")
        root = Path(unquote(uri.path)).resolve()
        manifest_path = root / "manifest.json"
        if reference.checksum is not None and _digest(manifest_path) != reference.checksum:
            raise ValueError("Checkpoint manifest checksum mismatch")
        manifest = CheckpointManifest.model_validate_json(manifest_path.read_bytes())
        if manifest.format != reference.format:
            raise ValueError("Checkpoint format differs from manifest")
        actual = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("Checkpoint bundles cannot contain symlinks")
            if path.is_file() and path != manifest_path:
                actual.add(path.relative_to(root).as_posix())
        if actual != set(manifest.files):
            raise ValueError("Checkpoint bundle files differ from manifest")
        for name, checksum in manifest.files.items():
            if _digest(_relative_file(root, name)) != checksum:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
        if manifest.entrypoint not in manifest.files:
            raise ValueError("Checkpoint entrypoint is not in manifest")
        return ResolvedCheckpoint(
            format=manifest.format,
            source_uri=reference.uri,
            local_path=_relative_file(root, manifest.entrypoint),
            metadata={**manifest.metadata, "resume_scope": manifest.resume_scope},
        )
