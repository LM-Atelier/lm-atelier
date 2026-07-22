from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .artifacts import ArtifactStore
from .config import Settings
from .domain import ArtifactKind
from .models import Artifact, Chat, Message, MessagePart, Project, Run
from .schemas import ChatDetail, ProjectOut, RunOut


class ProjectExporter:
    def __init__(self, settings: Settings, artifacts: ArtifactStore) -> None:
        self.settings = settings
        self.artifacts = artifacts

    def export(self, session: Session, project_id: str) -> Artifact:
        project = session.get(Project, project_id)
        if not project:
            raise LookupError("project not found")
        chats = session.scalars(
            select(Chat)
            .options(
                selectinload(Chat.messages)
                .selectinload(Message.parts)
                .selectinload(MessagePart.artifact)
            )
            .where(Chat.project_id == project_id)
            .order_by(Chat.created_at)
        ).all()
        runs = session.scalars(
            select(Run).where(Run.chat_id.in_([chat.id for chat in chats]))
        ).all()
        referenced: dict[str, Artifact] = {}
        for chat in chats:
            for message in chat.messages:
                for part in message.parts:
                    if part.artifact:
                        referenced[part.artifact.id] = part.artifact

        manifest = {
            "format": "local-lm-project",
            "version": 1,
            "project": ProjectOut.model_validate(project).model_dump(mode="json"),
            "chats": [ChatDetail.model_validate(chat).model_dump(mode="json") for chat in chats],
            "runs": [RunOut.model_validate(run).model_dump(mode="json") for run in runs],
            "artifacts": [
                {
                    "id": artifact.id,
                    "sha256": artifact.sha256,
                    "kind": artifact.kind,
                    "media_type": artifact.media_type,
                    "size_bytes": artifact.size_bytes,
                    "original_name": artifact.original_name,
                    "metadata": artifact.metadata_json,
                    "archive_path": self._archive_path(artifact),
                }
                for artifact in referenced.values()
            ],
        }
        with tempfile.NamedTemporaryFile(
            dir=self.settings.export_dir, suffix=".local-lm.zip", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
                for artifact in referenced.values():
                    archive.write(
                        self.artifacts.resolve(artifact),
                        self._archive_path(artifact),
                        compress_type=zipfile.ZIP_STORED,
                    )
            return self.artifacts.ingest_path(
                session,
                temporary,
                kind=ArtifactKind.EXPORT,
                media_type="application/zip",
                original_name=f"{self._safe_name(project.name)}.local-lm.zip",
                metadata={
                    "format": "local-lm-project",
                    "version": 1,
                    "project_id": project.id,
                    "artifact_count": len(referenced),
                },
            )
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = "".join(
            character if character.isalnum() or character in "-_" else "-" for character in value
        )
        return safe.strip("-")[:80] or "project"

    @staticmethod
    def _archive_path(artifact: Artifact) -> str:
        name = Path(artifact.original_name or artifact.sha256).name
        return f"artifacts/{artifact.sha256}/{name}"
