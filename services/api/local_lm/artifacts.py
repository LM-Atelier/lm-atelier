from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
import tempfile
from pathlib import Path
from typing import IO

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .domain import ArtifactKind
from .models import Artifact


class ArtifactStore:
    def __init__(self, settings: Settings) -> None:
        self.root = settings.artifact_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _destination(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:4] / digest

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def resolve(self, artifact: Artifact) -> Path:
        path = (self.root / artifact.relative_path).resolve()
        if self.root not in path.parents:
            raise ValueError("artifact path escapes store")
        return path

    def ingest_path(
        self,
        session: Session,
        source: Path,
        *,
        kind: ArtifactKind,
        media_type: str | None = None,
        original_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Artifact:
        source = source.resolve(strict=True)
        with source.open("rb") as handle:
            return self.ingest_stream(
                session,
                handle,
                kind=kind,
                media_type=media_type or mimetypes.guess_type(source.name)[0],
                original_name=original_name or source.name,
                metadata=metadata,
            )

    def ingest_bytes(
        self,
        session: Session,
        content: bytes,
        *,
        kind: ArtifactKind,
        media_type: str,
        original_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Artifact:
        with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as handle:
            handle.write(content)
            handle.seek(0)
            return self.ingest_stream(
                session,
                handle,
                kind=kind,
                media_type=media_type,
                original_name=original_name,
                metadata=metadata,
            )

    def ingest_stream(
        self,
        session: Session,
        source: IO[bytes],
        *,
        kind: ArtifactKind,
        media_type: str | None,
        original_name: str | None,
        metadata: dict[str, object] | None,
    ) -> Artifact:
        digest = hashlib.sha256()
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix="ingest-", dir=self.root)
        size = 0
        try:
            with os.fdopen(fd, "wb") as destination:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    destination.write(chunk)
                    size += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())

            sha256 = digest.hexdigest()
            existing = session.scalar(select(Artifact).where(Artifact.sha256 == sha256))
            if existing:
                return existing

            destination_path = self._destination(sha256)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if not destination_path.exists():
                os.replace(temporary_name, destination_path)
            artifact = Artifact(
                id=f"sha256:{sha256}",
                sha256=sha256,
                kind=kind.value,
                media_type=media_type or "application/octet-stream",
                size_bytes=size,
                relative_path=self._relative(destination_path),
                original_name=original_name,
                metadata_json=metadata or {},
            )
            session.add(artifact)
            session.flush()
            return artifact
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def export_copy(self, artifact: Artifact, destination: Path) -> Path:
        source = self.resolve(artifact)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination
