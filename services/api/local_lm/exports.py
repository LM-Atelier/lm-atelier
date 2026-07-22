from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO, Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .artifacts import ArtifactStore
from .config import Settings
from .domain import (
    ArtifactKind,
    MessageRole,
    MessageStatus,
    Operation,
    PartType,
    RoutingMode,
    RunStatus,
)
from .models import Artifact, Chat, Message, MessagePart, Project, Run
from .schemas import ChatDetail, ProjectOut, RunOut


class ProjectExporter:
    def __init__(self, settings: Settings, artifacts: ArtifactStore) -> None:
        self.settings = settings
        self.artifacts = artifacts

    def export(self, session: Session, project_id: str, *, include_media: bool = True) -> Artifact:
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
        poster_ids = {
            poster_id
            for artifact in referenced.values()
            if isinstance((poster_id := artifact.metadata_json.get("poster_artifact_id")), str)
        }
        if poster_ids:
            for artifact in session.scalars(select(Artifact).where(Artifact.id.in_(poster_ids))):
                referenced[artifact.id] = artifact

        manifest = {
            "format": "local-lm-project",
            "version": 2,
            "media_included": include_media,
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
                for artifact in sorted(referenced.values(), key=lambda item: item.id)
            ],
        }
        with tempfile.NamedTemporaryFile(
            dir=self.settings.export_dir, suffix=".lm-atelier.zip", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
                if include_media:
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
                original_name=f"{self._safe_name(project.name)}.lm-atelier.zip",
                metadata={
                    "format": "local-lm-project",
                    "version": 2,
                    "project_id": project.id,
                    "artifact_count": len(referenced),
                    "media_included": include_media,
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

    def import_archive(self, session: Session, source: IO[bytes]) -> Project:
        source.seek(0)
        try:
            with zipfile.ZipFile(source) as archive:
                infos = self._validate_archive(archive)
                manifest_info = infos.get("manifest.json")
                if not manifest_info or manifest_info.file_size > 50 * 1024 * 1024:
                    raise ValueError("project archive has no valid manifest")
                manifest = json.loads(archive.read(manifest_info))
                self._validate_manifest(manifest)
                artifact_map = self._import_artifacts(session, archive, infos, manifest)
                return self._import_records(session, manifest, artifact_map)
        except (json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise ValueError("invalid LM Atelier project archive") from exc

    def _validate_archive(self, archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
        infos = archive.infolist()
        if len(infos) > self.settings.max_project_archive_entries:
            raise ValueError("project archive contains too many entries")
        total_size = 0
        validated: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError("project archive contains an unsafe path")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("project archive cannot contain symbolic links")
            total_size += info.file_size
            if total_size > self.settings.max_project_import_bytes:
                raise ValueError("project archive expands beyond the configured limit")
            if not info.is_dir():
                normalized = str(path)
                if normalized in validated:
                    raise ValueError("project archive contains duplicate paths")
                validated[normalized] = info
        return validated

    @staticmethod
    def _validate_manifest(manifest: Any) -> None:
        if not isinstance(manifest, dict):
            raise ValueError("project manifest must be an object")
        if manifest.get("format") != "local-lm-project" or manifest.get("version") not in {
            1,
            2,
        }:
            raise ValueError("unsupported project archive format")
        if not isinstance(manifest.get("project"), dict):
            raise ValueError("project manifest is missing project metadata")
        for key, maximum in (("chats", 10_000), ("runs", 100_000), ("artifacts", 100_000)):
            value = manifest.get(key)
            if not isinstance(value, list) or len(value) > maximum:
                raise ValueError(f"project manifest has invalid {key}")

    def _import_artifacts(
        self,
        session: Session,
        archive: zipfile.ZipFile,
        infos: dict[str, zipfile.ZipInfo],
        manifest: dict[str, Any],
    ) -> dict[str, str]:
        imported: dict[str, str] = {}
        seen: set[str] = set()
        for record in manifest["artifacts"]:
            if not isinstance(record, dict):
                raise ValueError("project manifest has an invalid artifact record")
            old_id = self._text(record.get("id"), "artifact id", 80)
            if old_id in seen:
                raise ValueError("project manifest contains duplicate artifact ids")
            seen.add(old_id)
            digest = self._text(record.get("sha256"), "artifact checksum", 64)
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("project manifest has an invalid artifact checksum")
            archive_path = self._text(record.get("archive_path"), "artifact path", 1000)
            path_parts = PurePosixPath(archive_path).parts
            if len(path_parts) < 3 or path_parts[:2] != ("artifacts", digest):
                raise ValueError("project manifest has an invalid artifact path")
            info = infos.get(archive_path)
            if not info:
                continue
            expected_size = record.get("size_bytes")
            if (
                not isinstance(expected_size, int)
                or expected_size < 0
                or info.file_size != expected_size
            ):
                raise ValueError("project artifact size does not match its manifest")
            with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as staged:
                checksum = hashlib.sha256()
                size = 0
                with archive.open(info) as extracted:
                    while chunk := extracted.read(1024 * 1024):
                        checksum.update(chunk)
                        staged.write(chunk)
                        size += len(chunk)
                if size != expected_size or checksum.hexdigest() != digest:
                    raise ValueError("project artifact checksum does not match its manifest")
                staged.seek(0)
                try:
                    kind = ArtifactKind(str(record.get("kind")))
                except ValueError:
                    kind = ArtifactKind.OTHER
                metadata = record.get("metadata")
                artifact = self.artifacts.ingest_stream(
                    session,
                    staged,
                    kind=kind,
                    media_type=self._text(record.get("media_type"), "media type", 120),
                    original_name=self._optional_text(record.get("original_name"), 500),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            imported[old_id] = artifact.id
        return imported

    def _import_records(
        self,
        session: Session,
        manifest: dict[str, Any],
        artifacts: dict[str, str],
    ) -> Project:
        project_data = manifest["project"]
        project = Project(
            name=self._text(project_data.get("name"), "project name", 200),
            description=self._optional_text(project_data.get("description"), 10_000) or "",
            instructions=self._optional_text(project_data.get("instructions"), 100_000) or "",
            archived=bool(project_data.get("archived", False)),
        )
        session.add(project)
        session.flush()
        chat_map: dict[str, Chat] = {}
        message_map: dict[str, Message] = {}
        chat_records: dict[str, dict[str, Any]] = {}
        for chat_data in manifest["chats"]:
            if not isinstance(chat_data, dict):
                raise ValueError("project manifest has an invalid chat")
            old_chat_id = self._text(chat_data.get("id"), "chat id", 40)
            if old_chat_id in chat_map:
                raise ValueError("project manifest contains duplicate chat ids")
            chat = Chat(
                project_id=project.id,
                title=self._text(chat_data.get("title"), "chat title", 240),
                archived=bool(chat_data.get("archived", False)),
                routing_mode=RoutingMode(str(chat_data.get("routing_mode", "auto"))).value,
                confirm_uncertain_media=bool(chat_data.get("confirm_uncertain_media", True)),
            )
            session.add(chat)
            session.flush()
            chat_map[old_chat_id] = chat
            chat_records[old_chat_id] = chat_data
            messages = chat_data.get("messages")
            if not isinstance(messages, list) or len(messages) > 100_000:
                raise ValueError("project manifest has invalid messages")
            for message_data in messages:
                if not isinstance(message_data, dict):
                    raise ValueError("project manifest has an invalid message")
                old_message_id = self._text(message_data.get("id"), "message id", 40)
                if old_message_id in message_map:
                    raise ValueError("project manifest contains duplicate message ids")
                message = Message(
                    chat_id=chat.id,
                    role=MessageRole(str(message_data.get("role"))).value,
                    status=MessageStatus(str(message_data.get("status"))).value,
                )
                session.add(message)
                session.flush()
                message_map[old_message_id] = message
                parts = message_data.get("parts")
                if not isinstance(parts, list) or len(parts) > 10_000:
                    raise ValueError("project manifest has invalid message parts")
                for part_data in parts:
                    if not isinstance(part_data, dict):
                        raise ValueError("project manifest has an invalid message part")
                    old_artifact_id = part_data.get("artifact_id")
                    metadata = part_data.get("metadata_json")
                    metadata = dict(metadata) if isinstance(metadata, dict) else {}
                    artifact_id = (
                        artifacts.get(old_artifact_id) if isinstance(old_artifact_id, str) else None
                    )
                    if old_artifact_id and not artifact_id:
                        metadata["missing_import_artifact_id"] = old_artifact_id
                    message.parts.append(
                        MessagePart(
                            position=len(message.parts),
                            type=PartType(str(part_data.get("type"))).value,
                            text=self._optional_text(part_data.get("text"), 10_000_000),
                            artifact_id=artifact_id,
                            metadata_json=metadata,
                        )
                    )
        for old_chat_id, chat in chat_map.items():
            data = chat_records[old_chat_id]
            for message_data in data["messages"]:
                message = message_map[str(message_data["id"])]
                parent_id = message_data.get("parent_id")
                message.parent_id = (
                    message_map[parent_id].id
                    if isinstance(parent_id, str) and parent_id in message_map
                    else None
                )
            active_head = data.get("active_head_message_id")
            chat.active_head_message_id = (
                message_map[active_head].id
                if isinstance(active_head, str) and active_head in message_map
                else None
            )
        for run_data in manifest["runs"]:
            if not isinstance(run_data, dict):
                raise ValueError("project manifest has an invalid run")
            imported_chat = chat_map.get(str(run_data.get("chat_id")))
            user_message = message_map.get(str(run_data.get("user_message_id")))
            assistant_message = message_map.get(str(run_data.get("assistant_message_id")))
            if not imported_chat or not user_message or not assistant_message:
                raise ValueError("project run references a missing chat or message")
            status = RunStatus(str(run_data.get("status", "failed")))
            interrupted = status not in {
                RunStatus.COMPLETE,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }
            provenance = run_data.get("provenance_json")
            provenance = dict(provenance) if isinstance(provenance, dict) else {}
            provenance["imported_from_run_id"] = run_data.get("id")
            session.add(
                Run(
                    chat_id=imported_chat.id,
                    user_message_id=user_message.id,
                    assistant_message_id=assistant_message.id,
                    operation=Operation(str(run_data.get("operation"))).value,
                    status=RunStatus.FAILED.value if interrupted else status.value,
                    standalone_prompt=self._optional_text(
                        run_data.get("standalone_prompt"), 10_000_000
                    )
                    or "",
                    settings_json=run_data.get("settings_json")
                    if isinstance(run_data.get("settings_json"), dict)
                    else {},
                    provenance_json=provenance,
                    error="Imported while generation was incomplete."
                    if interrupted
                    else self._optional_text(run_data.get("error"), 1_000_000),
                )
            )
        session.flush()
        return project

    @staticmethod
    def _text(value: object, label: str, maximum: int) -> str:
        if not isinstance(value, str) or not value or len(value) > maximum:
            raise ValueError(f"project manifest has an invalid {label}")
        return value

    @staticmethod
    def _optional_text(value: object, maximum: int) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or len(value) > maximum:
            raise ValueError("project manifest contains invalid text")
        return value
