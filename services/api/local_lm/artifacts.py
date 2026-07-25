from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import shutil
import tempfile
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import Settings
from .domain import ArtifactKind
from .models import Artifact, MessagePart


class ArtifactStore:
    def __init__(self, settings: Settings) -> None:
        self.root = settings.artifact_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _destination(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:4] / digest

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

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
                if existing.metadata_json.get("temporary_preview") and not (metadata or {}).get(
                    "temporary_preview"
                ):
                    existing.kind = kind.value
                    existing.media_type = media_type or existing.media_type
                    existing.original_name = original_name or existing.original_name
                    existing.metadata_json = metadata or {}
                    session.flush()
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

    async def video_poster(self, artifact: Artifact) -> bytes | None:
        executable = shutil.which("ffmpeg")
        if not executable or not artifact.media_type.startswith("video/"):
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(self.resolve(artifact)),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except (OSError, TimeoutError):
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            return None
        return stdout if process.returncode == 0 and stdout else None

    async def browser_video_proxy(self, artifact: Artifact) -> tuple[bytes, str, str] | None:
        if artifact.media_type in {"video/mp4", "video/webm"}:
            return None
        executable = shutil.which("ffmpeg")
        if not executable:
            return None
        fd, temporary_name = tempfile.mkstemp(prefix="video-proxy-", suffix=".mp4", dir=self.root)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(self.resolve(artifact)),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(temporary),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=600)
            if process.returncode or not temporary.is_file() or temporary.stat().st_size == 0:
                return None
            return (
                temporary.read_bytes(),
                "video/mp4",
                f"{artifact.original_name or 'video'}.proxy.mp4",
            )
        except (OSError, TimeoutError):
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            return None
        finally:
            temporary.unlink(missing_ok=True)

    def delete_temporary_preview(self, session: Session, artifact_id: str) -> bool:
        artifact = session.get(Artifact, artifact_id)
        if not artifact or not artifact.metadata_json.get("temporary_preview"):
            return False
        references = (
            session.scalar(
                select(func.count(MessagePart.id)).where(MessagePart.artifact_id == artifact_id)
            )
            or 0
        )
        if references:
            return False
        path = self.resolve(artifact)
        session.delete(artifact)
        session.flush()
        path.unlink(missing_ok=True)
        for parent in (path.parent, path.parent.parent):
            with suppress(OSError):
                parent.rmdir()
        return True

    @staticmethod
    def referenced_artifact_ids(session: Session) -> set[str]:
        referenced = {
            artifact_id
            for artifact_id in session.scalars(
                select(MessagePart.artifact_id).where(MessagePart.artifact_id.is_not(None))
            )
            if artifact_id
        }
        if not referenced:
            return referenced
        posters = session.scalars(select(Artifact).where(Artifact.id.in_(referenced))).all()
        referenced.update(
            linked_id
            for artifact in posters
            for key in ("poster_artifact_id", "browser_proxy_artifact_id")
            if isinstance((linked_id := artifact.metadata_json.get(key)), str)
        )
        return referenced

    def cleanup_retention(
        self,
        session: Session,
        *,
        retention_days: int,
        temporary_hours: int,
        dry_run: bool,
        now: datetime | None = None,
    ) -> tuple[int, int, int]:
        current = now or datetime.now(UTC)
        referenced = self.referenced_artifact_ids(session)
        marked_count = 0
        removed_count = 0
        reclaimed_bytes = 0
        for artifact in session.scalars(select(Artifact).order_by(Artifact.created_at)).all():
            metadata = dict(artifact.metadata_json)
            if artifact.id in referenced:
                if "unreferenced_at" in metadata and not dry_run:
                    metadata.pop("unreferenced_at", None)
                    artifact.metadata_json = metadata
                continue
            temporary = bool(metadata.get("temporary_preview") or metadata.get("intermediate"))
            age = current - self._aware(artifact.created_at)
            eligible = temporary and age >= timedelta(hours=temporary_hours)
            unreferenced_at = self._metadata_datetime(metadata.get("unreferenced_at"))
            if not temporary and unreferenced_at:
                eligible = current - unreferenced_at >= timedelta(days=retention_days)
            if eligible:
                removed_count += 1
                reclaimed_bytes += artifact.size_bytes
                if not dry_run:
                    self._delete_artifact(session, artifact)
                continue
            if not temporary and not unreferenced_at:
                marked_count += 1
                if not dry_run:
                    metadata["unreferenced_at"] = current.isoformat()
                    artifact.metadata_json = metadata
        if not dry_run:
            session.flush()
        return marked_count, removed_count, reclaimed_bytes

    def delete_library_artifact(
        self,
        session: Session,
        artifact: Artifact,
    ) -> tuple[int, int, int]:
        if artifact.kind not in {ArtifactKind.IMAGE.value, ArtifactKind.VIDEO.value}:
            raise ValueError("only image and video library artifacts can be deleted directly")

        linked_ids = {
            linked_id
            for key in ("poster_artifact_id", "browser_proxy_artifact_id")
            if isinstance((linked_id := artifact.metadata_json.get(key)), str)
        }
        parts = session.scalars(
            select(MessagePart).where(MessagePart.artifact_id == artifact.id)
        ).all()
        for part in parts:
            part.artifact_id = None
        session.flush()

        removed_count = 1
        reclaimed_bytes = artifact.size_bytes
        self._delete_artifact(session, artifact)

        referenced = self.referenced_artifact_ids(session)
        for linked_id in linked_ids:
            linked = session.get(Artifact, linked_id)
            if not linked or linked.id in referenced:
                continue
            removed_count += 1
            reclaimed_bytes += linked.size_bytes
            self._delete_artifact(session, linked)
        return len(parts), removed_count, reclaimed_bytes

    def _delete_artifact(self, session: Session, artifact: Artifact) -> None:
        path = self.resolve(artifact)
        session.delete(artifact)
        session.flush()
        path.unlink(missing_ok=True)
        for parent in (path.parent, path.parent.parent):
            with suppress(OSError):
                parent.rmdir()

    @staticmethod
    def _aware(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @classmethod
    def _metadata_datetime(cls, value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        with suppress(ValueError):
            return cls._aware(datetime.fromisoformat(value))
        return None
