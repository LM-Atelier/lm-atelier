from __future__ import annotations

import hashlib
import json
import math
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import IO, Any, cast

from pydantic import ValidationError
from sqlalchemy import event, select
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
from .models import (
    Artifact,
    Chat,
    GenerationPreset,
    Message,
    MessagePart,
    ModelAssetInstall,
    ModelSource,
    Project,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
)
from .profile_service import AUTO_PROFILE_ID
from .project_dependencies import (
    DependencySourceIndex,
    ImportedDependencies,
    build_dependency_manifest,
    dependency_source_index,
    install_dependency_manifest,
    parse_dependency_manifest,
)
from .project_portability import has_local_path, redact_local_paths
from .prompt_helpers import STANDARD_CHAT_SCOPE
from .schemas import ChatDetail, ProjectOut, RunOut, SettingField, VisionSettings
from .settings_registry import validate_settings

_CAS_IMPORT_SESSION_KEY = "lm_atelier_project_import_cas"
_AUXILIARY_ASSET_KINDS = {
    "lora",
    "vae",
    "controlnet",
    "upscaler",
    "embedding",
    "ip_adapter",
}


class _ImportCasTransaction:
    def __init__(self, session: Session, root: Path) -> None:
        self._session = session
        self._root = root.resolve()
        self._created_paths: set[Path] = set()
        self._registered = False
        self._resolved = False

    def track_created(self, path: Path, *, existed_before: bool) -> None:
        if existed_before or not path.is_file():
            return
        resolved = path.resolve()
        if self._root not in resolved.parents:
            raise ValueError("project artifact destination escapes the content store")
        self._created_paths.add(resolved)
        if self._registered:
            return
        self._registered = True
        self._session.info[_CAS_IMPORT_SESSION_KEY] = self
        event.listen(self._session, "after_commit", self._after_commit)
        event.listen(self._session, "after_rollback", self._after_rollback)
        event.listen(self._session, "after_transaction_end", self._after_transaction_end)

    def rollback(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        for path in self._created_paths:
            path.unlink(missing_ok=True)
        self._created_paths.clear()
        self._session.info.pop(_CAS_IMPORT_SESSION_KEY, None)

    def _after_commit(self, _session: Session) -> None:
        self._resolved = True
        self._created_paths.clear()
        self._session.info.pop(_CAS_IMPORT_SESSION_KEY, None)

    def _after_rollback(self, _session: Session) -> None:
        self.rollback()

    def _after_transaction_end(self, _session: Session, transaction: Any) -> None:
        if transaction.parent is None and not self._resolved:
            self.rollback()


class ProjectExporter:
    def __init__(self, settings: Settings, artifacts: ArtifactStore) -> None:
        self.settings = settings
        self.artifacts = artifacts
        # Live engine schema per role, supplied per import by the request layer.
        self._known_fields: dict[str, list[SettingField]] = {}

    def export(self, session: Session, project_id: str, *, include_media: bool = True) -> Artifact:
        project = session.get(Project, project_id)
        if not project:
            raise LookupError("project not found")
        chats = session.scalars(
            select(Chat)
            .options(
                selectinload(Chat.messages)
                .selectinload(Message.parts)
                .selectinload(MessagePart.artifact),
                selectinload(Chat.messages)
                .selectinload(Message.response_revisions)
                .selectinload(ResponseRevision.parts)
                .selectinload(ResponseRevisionPart.artifact),
            )
            .where(Chat.project_id == project_id, Chat.scope == STANDARD_CHAT_SCOPE)
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
                for revision in message.response_revisions:
                    for revision_part in revision.parts:
                        if revision_part.artifact:
                            referenced[revision_part.artifact.id] = revision_part.artifact
        run_input_ids: set[str] = set()
        for run in runs:
            provenance = run.provenance_json if isinstance(run.provenance_json, dict) else {}
            input_ids = provenance.get("input_artifact_ids")
            if isinstance(input_ids, list):
                run_input_ids.update(
                    artifact_id for artifact_id in input_ids if isinstance(artifact_id, str)
                )
        if run_input_ids:
            for artifact in session.scalars(select(Artifact).where(Artifact.id.in_(run_input_ids))):
                referenced[artifact.id] = artifact
        linked_artifact_ids = {
            linked_id
            for artifact in referenced.values()
            for key in ("poster_artifact_id", "browser_proxy_artifact_id")
            if isinstance((linked_id := artifact.metadata_json.get(key)), str)
        }
        if linked_artifact_ids:
            for artifact in session.scalars(
                select(Artifact).where(Artifact.id.in_(linked_artifact_ids))
            ):
                referenced[artifact.id] = artifact

        dependency_manifest, dependency_index = build_dependency_manifest(
            session, project, list(chats), list(runs)
        )
        project_record = ProjectOut.model_validate(project).model_dump(mode="json")
        self._snapshot_generation_defaults(
            session,
            project,
            project_record,
            dependency_index,
        )
        project_record["generation_settings_json"] = redact_local_paths(
            project_record["generation_settings_json"]
        )
        project_record["image_workflow_revision_id"] = self._portable_revision_reference(
            project.image_workflow_revision_id,
            dependency_index,
            {Operation.TEXT_TO_IMAGE.value, Operation.IMAGE_TO_IMAGE.value},
        )
        project_record["video_workflow_revision_id"] = self._portable_revision_reference(
            project.video_workflow_revision_id,
            dependency_index,
            {Operation.TEXT_TO_VIDEO.value, Operation.IMAGE_TO_VIDEO.value},
        )
        runs_by_id = {run.id: run for run in runs}
        chat_records: list[dict[str, Any]] = []
        for chat in chats:
            record = ChatDetail.model_validate(chat).model_dump(mode="json")
            self._snapshot_generation_defaults(session, chat, record, dependency_index)
            record["generation_settings_json"] = redact_local_paths(
                record["generation_settings_json"]
            )
            for field, role in (
                ("active_chat_profile_id", "chat"),
                ("active_vision_profile_id", "chat"),
                ("active_image_profile_id", "image"),
                ("active_video_profile_id", "video"),
            ):
                record[field] = self._portable_profile_reference(
                    getattr(chat, field), dependency_index, role
                )
            self._sanitize_exported_message_metadata(
                record,
                runs_by_id,
                dependency_index,
                set(referenced),
            )
            chat_records.append(record)
        run_records: list[dict[str, Any]] = []
        for run in runs:
            record = RunOut.model_validate(run).model_dump(mode="json")
            operation = Operation(run.operation)
            record["profile_id"] = self._portable_profile_reference(
                run.profile_id,
                dependency_index,
                self._role_for_operation(operation),
                allow_auto=False,
            )
            record["vision_profile_id"] = self._portable_profile_reference(
                run.vision_profile_id,
                dependency_index,
                "chat",
                allow_auto=False,
            )
            record["workflow_revision_id"] = self._portable_revision_reference(
                run.workflow_revision_id,
                dependency_index,
                {operation.value},
            )
            record["settings_json"] = redact_local_paths(record["settings_json"])
            record["provenance_json"] = self._portable_provenance(
                record["provenance_json"],
                operation,
                dependency_index,
                set(referenced),
            )
            record["error"] = redact_local_paths(record["error"])
            run_records.append(record)
        auxiliary_requirements, auxiliary_references = self._auxiliary_requirements(
            session,
            [project_record, *chat_records, *run_records],
        )
        for record in [project_record, *chat_records, *run_records]:
            self._remap_auxiliary_asset_references(record, auxiliary_references)
        manifest = {
            "format": "local-lm-project",
            "version": 6,
            "media_included": include_media,
            "project": project_record,
            "chats": chat_records,
            "runs": run_records,
            "artifacts": [
                {
                    "id": artifact.id,
                    "sha256": artifact.sha256,
                    "kind": artifact.kind,
                    "media_type": artifact.media_type,
                    "size_bytes": artifact.size_bytes,
                    "original_name": redact_local_paths(artifact.original_name),
                    "metadata": self._portable_artifact_metadata(
                        artifact.metadata_json,
                        set(referenced),
                    ),
                    "archive_path": self._archive_path(artifact),
                }
                for artifact in sorted(referenced.values(), key=lambda item: item.id)
            ],
            "dependencies": dependency_manifest,
            "auxiliary_requirements": auxiliary_requirements,
        }
        if has_local_path([record["provenance_json"] for record in run_records]):
            raise ValueError("project export contains a non-portable local path")
        with tempfile.NamedTemporaryFile(
            dir=self.settings.export_dir, suffix=".lm-atelier.zip", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
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
                    "version": 6,
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
        raw_name = str(artifact.original_name or artifact.sha256).replace("\\", "/")
        name = PurePosixPath(raw_name).name
        if name in {"", ".", ".."}:
            name = artifact.sha256
        return f"artifacts/{artifact.sha256}/{name}"

    @staticmethod
    def _remap_auxiliary_asset_references(
        value: object,
        mappings: dict[str, str],
    ) -> None:
        stack = [value]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                asset_id = current.get("asset_id")
                if isinstance(asset_id, str) and asset_id in mappings:
                    current["asset_id"] = mappings[asset_id]
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)

    @classmethod
    def _auxiliary_requirements(
        cls,
        session: Session,
        records: list[object],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        referenced_ids: set[str] = set()
        stack = list(records)
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                asset_id = current.get("asset_id")
                if isinstance(asset_id, str):
                    referenced_ids.add(asset_id)
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
        if not referenced_ids:
            return [], {}

        assets = session.scalars(
            select(ModelAssetInstall).where(ModelAssetInstall.id.in_(referenced_ids))
        ).all()
        source_ids = {asset.source_id for asset in assets if asset.source_id}
        sources = {
            source.id: source
            for source in (
                session.scalars(select(ModelSource).where(ModelSource.id.in_(source_ids))).all()
                if source_ids
                else []
            )
        }
        requirements: dict[str, dict[str, Any]] = {}
        mappings: dict[str, str] = {}
        for asset in sorted(assets, key=lambda item: item.id):
            digest = asset.manifest_json.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"auxiliary asset {asset.name!r} has no immutable verified checksum"
                )
            if asset.kind not in _AUXILIARY_ASSET_KINDS:
                raise ValueError(f"auxiliary asset {asset.name!r} has an unsupported kind")
            reference = f"auxiliary:{asset.kind}:sha256:{digest}"
            source = sources.get(asset.source_id or "")
            metadata = asset.manifest_json.get("metadata")
            hints: dict[str, Any] = {}
            if isinstance(metadata, dict):
                network_type = metadata.get("network_type")
                rank = metadata.get("rank")
                trigger_words = metadata.get("trigger_words")
                if isinstance(network_type, str):
                    hints["network_type"] = network_type[:120]
                if isinstance(rank, int) and not isinstance(rank, bool) and 0 < rank <= 1_000_000:
                    hints["rank"] = rank
                if isinstance(trigger_words, list):
                    hints["trigger_words"] = [
                        word[:200] for word in trigger_words[:100] if isinstance(word, str) and word
                    ]
            requirement: dict[str, Any] = {
                "id": reference,
                "kind": asset.kind,
                "name": asset.name[:300],
                "family": asset.family,
                "sha256": digest,
                "size_bytes": asset.size_bytes,
                "metadata": hints,
            }
            if source:
                requirement["source"] = {
                    "provider": source.provider,
                    "remote_id": source.remote_id,
                    "revision": source.revision,
                }
            requirements.setdefault(reference, requirement)
            mappings[asset.id] = reference
        return [requirements[key] for key in sorted(requirements)], mappings

    @staticmethod
    def _validate_auxiliary_requirements(value: object) -> None:
        if not isinstance(value, list) or len(value) > 1_000:
            raise ValueError("project manifest has invalid auxiliary requirements")
        seen_ids: set[str] = set()
        seen_hashes: set[tuple[str, str]] = set()
        for requirement in value:
            if not isinstance(requirement, dict):
                raise ValueError("project manifest has an invalid auxiliary requirement")
            kind = requirement.get("kind")
            digest = requirement.get("sha256")
            reference = requirement.get("id")
            if (
                not isinstance(kind, str)
                or kind not in _AUXILIARY_ASSET_KINDS
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or reference != f"auxiliary:{kind}:sha256:{digest}"
            ):
                raise ValueError("project manifest has an invalid auxiliary requirement identity")
            if reference in seen_ids or (kind, digest) in seen_hashes:
                raise ValueError("project manifest has duplicate auxiliary requirements")
            seen_ids.add(reference)
            seen_hashes.add((kind, digest))
            name = requirement.get("name")
            family = requirement.get("family")
            size_bytes = requirement.get("size_bytes")
            metadata = requirement.get("metadata")
            if (
                not isinstance(name, str)
                or not name
                or len(name) > 300
                or (family is not None and (not isinstance(family, str) or len(family) > 100))
                or not isinstance(size_bytes, int)
                or isinstance(size_bytes, bool)
                or size_bytes < 0
                or not isinstance(metadata, dict)
            ):
                raise ValueError("project manifest has invalid auxiliary requirement metadata")
            source = requirement.get("source")
            if source is not None and (
                not isinstance(source, dict)
                or not isinstance(source.get("provider"), str)
                or not isinstance(source.get("remote_id"), str)
                or not isinstance(source.get("revision"), str)
                or len(source["provider"]) > 32
                or len(source["remote_id"]) > 500
                or len(source["revision"]) > 200
            ):
                raise ValueError("project manifest has an invalid auxiliary requirement source")

    @staticmethod
    def _resolve_auxiliary_requirements(
        session: Session,
        manifest: dict[str, Any],
    ) -> dict[str, str]:
        if manifest["version"] < 6:
            return {}
        requirements = manifest["auxiliary_requirements"]
        identities = {
            (str(requirement["kind"]), str(requirement["sha256"])): str(requirement["id"])
            for requirement in requirements
        }
        if not identities:
            return {}
        candidates = session.scalars(
            select(ModelAssetInstall).where(
                ModelAssetInstall.kind.in_({kind for kind, _digest in identities}),
                ModelAssetInstall.verified_at.is_not(None),
            )
        ).all()
        resolved: dict[str, ModelAssetInstall] = {}
        for candidate in candidates:
            digest = candidate.manifest_json.get("sha256")
            if not isinstance(digest, str):
                continue
            reference = identities.get((candidate.kind, digest))
            if not reference:
                continue
            existing = resolved.get(reference)
            if existing is None or (candidate.active and not existing.active):
                resolved[reference] = candidate
        return {reference: asset.id for reference, asset in resolved.items()}

    def _sanitize_exported_message_metadata(
        self,
        chat_record: dict[str, Any],
        runs: dict[str, Run],
        dependencies: DependencySourceIndex,
        artifact_ids: set[str],
    ) -> None:
        messages = chat_record.get("messages")
        if not isinstance(messages, list):
            return
        for message in messages:
            if not isinstance(message, dict):
                continue
            part_groups: list[list[Any]] = []
            if isinstance(message.get("parts"), list):
                part_groups.append(message["parts"])
            revisions = message.get("response_revisions")
            if isinstance(revisions, list):
                part_groups.extend(
                    revision["parts"]
                    for revision in revisions
                    if isinstance(revision, dict) and isinstance(revision.get("parts"), list)
                )
            for parts in part_groups:
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    embedded_artifact = part.get("artifact")
                    if isinstance(embedded_artifact, dict):
                        embedded_artifact["original_name"] = redact_local_paths(
                            embedded_artifact.get("original_name")
                        )
                        embedded_artifact["metadata_json"] = self._portable_artifact_metadata(
                            embedded_artifact.get("metadata_json"),
                            artifact_ids,
                        )
                    metadata = part.get("metadata_json")
                    metadata = redact_local_paths(metadata) if isinstance(metadata, dict) else {}
                    source_run = runs.get(str(metadata.get("run_id")))
                    if source_run and isinstance(metadata.get("provenance"), dict):
                        metadata["provenance"] = self._portable_provenance(
                            metadata["provenance"],
                            Operation(source_run.operation),
                            dependencies,
                            artifact_ids,
                        )
                    metadata, _missing_count = self._remap_artifact_references(
                        metadata,
                        {artifact_id: artifact_id for artifact_id in artifact_ids},
                        artifact_ids,
                        strict=False,
                    )
                    part["metadata_json"] = metadata

    def _portable_artifact_metadata(
        self,
        value: object,
        artifact_ids: set[str],
    ) -> dict[str, Any]:
        metadata = redact_local_paths(value) if isinstance(value, dict) else {}
        mapped, _missing_count = self._remap_artifact_references(
            metadata,
            {artifact_id: artifact_id for artifact_id in artifact_ids},
            artifact_ids,
            strict=False,
        )
        return mapped

    def _portable_provenance(
        self,
        value: object,
        operation: Operation,
        dependencies: DependencySourceIndex,
        artifact_ids: set[str],
    ) -> dict[str, Any]:
        provenance = redact_local_paths(value) if isinstance(value, dict) else {}
        role = self._role_for_operation(operation)
        provenance, missing_artifact_count = self._remap_artifact_references(
            provenance,
            {artifact_id: artifact_id for artifact_id in artifact_ids},
            artifact_ids,
            strict=False,
        )

        model = provenance.get("model")
        if isinstance(model, dict):
            model.pop("install_id", None)
            model.pop("local_path", None)
            model["profile_id"] = self._portable_profile_reference(
                model.get("profile_id"),
                dependencies,
                role,
                allow_auto=False,
            )

        selection = provenance.get("model_selection")
        if isinstance(selection, dict):
            selection["profile_id"] = self._portable_profile_reference(
                selection.get("profile_id"),
                dependencies,
                role,
                allow_auto=False,
            )

        self._portable_vision_provenance(provenance, dependencies)

        preset = provenance.get("preset")
        if isinstance(preset, dict):
            preset["id"] = self._portable_preset_reference(
                preset.get("id"),
                dependencies,
                role,
            )
        layers = provenance.get("preset_layers")
        if isinstance(layers, list):
            for layer in layers:
                if isinstance(layer, dict):
                    layer["id"] = self._portable_preset_reference(
                        layer.get("id"),
                        dependencies,
                        role,
                    )

        workflow = provenance.get("workflow")
        if isinstance(workflow, dict):
            source_revision_id = self._portable_revision_reference(
                workflow.get("revision_id"),
                dependencies,
                {operation.value},
            )
            workflow["revision_id"] = source_revision_id
            workflow["definition_id"] = (
                dependencies.revision_workflow_ids.get(source_revision_id)
                if source_revision_id
                else None
            )
            workflow["trusted"] = False

        worker = provenance.get("worker")
        if isinstance(worker, dict):
            for local_only_key in (
                "command",
                "failure_detail",
                "log_path",
                "pid",
                "stderr_tail",
            ):
                worker.pop(local_only_key, None)

        provenance["portable_schema_version"] = 2
        if missing_artifact_count:
            provenance["unavailable_artifact_count"] = missing_artifact_count
        else:
            provenance.pop("unavailable_artifact_count", None)
        if has_local_path(provenance):
            raise ValueError("run provenance contains a non-portable local path")
        return provenance

    def _import_provenance(
        self,
        value: object,
        operation: Operation,
        dependencies: ImportedDependencies | None,
        artifact_mappings: dict[str, str],
        declared_artifact_ids: set[str],
        *,
        strict: bool,
    ) -> dict[str, Any]:
        provenance = redact_local_paths(value) if isinstance(value, dict) else {}
        provenance, missing_artifact_count = self._remap_artifact_references(
            provenance,
            artifact_mappings,
            declared_artifact_ids,
            strict=strict,
        )
        role = self._role_for_operation(operation)

        model = provenance.get("model")
        if isinstance(model, dict):
            model.pop("install_id", None)
            model.pop("local_path", None)
            model["profile_id"] = self._import_profile_reference(
                None,
                dependencies,
                model.get("profile_id"),
                role,
                allow_auto=False,
            )

        selection = provenance.get("model_selection")
        if isinstance(selection, dict):
            selection["profile_id"] = self._import_profile_reference(
                None,
                dependencies,
                selection.get("profile_id"),
                role,
                allow_auto=False,
            )

        self._import_vision_provenance(provenance, dependencies)

        preset = provenance.get("preset")
        if isinstance(preset, dict):
            preset["id"] = self._import_preset_reference(
                dependencies,
                preset.get("id"),
                role,
            )
        layers = provenance.get("preset_layers")
        if isinstance(layers, list):
            for layer in layers:
                if isinstance(layer, dict):
                    layer["id"] = self._import_preset_reference(
                        dependencies,
                        layer.get("id"),
                        role,
                    )

        workflow = provenance.get("workflow")
        if isinstance(workflow, dict):
            source_revision_id = workflow.get("revision_id")
            if dependencies and source_revision_id is not None:
                imported_revision_id = dependencies.revision(
                    source_revision_id,
                    {operation.value},
                )
                expected_workflow_source_id = dependencies.revision_workflow_source_ids.get(
                    str(source_revision_id)
                )
                supplied_workflow_source_id = workflow.get("definition_id")
                if (
                    supplied_workflow_source_id is not None
                    and supplied_workflow_source_id != expected_workflow_source_id
                ):
                    raise ValueError("project provenance has mismatched workflow identifiers")
                workflow["revision_id"] = imported_revision_id
                workflow["definition_id"] = dependencies.workflow(expected_workflow_source_id)
            else:
                workflow["revision_id"] = None
                workflow["definition_id"] = None
            workflow["trusted"] = False

        worker = provenance.get("worker")
        if isinstance(worker, dict):
            for local_only_key in (
                "command",
                "failure_detail",
                "log_path",
                "pid",
                "stderr_tail",
            ):
                worker.pop(local_only_key, None)

        provenance["portable_schema_version"] = 2
        if missing_artifact_count:
            provenance["unavailable_artifact_count"] = missing_artifact_count
        else:
            provenance.pop("unavailable_artifact_count", None)
        if has_local_path(provenance):
            raise ValueError("imported run provenance contains a local path")
        return provenance

    def _portable_vision_provenance(
        self,
        provenance: dict[str, Any],
        dependencies: DependencySourceIndex,
    ) -> None:
        context = provenance.get("context")
        vision = context.get("vision") if isinstance(context, dict) else None
        if not isinstance(vision, dict):
            return
        vision["profile_id"] = self._portable_profile_reference(
            vision.get("profile_id"),
            dependencies,
            "chat",
            allow_auto=False,
        )
        profile = vision.get("profile")
        if isinstance(profile, dict):
            profile.pop("install_id", None)
            profile["profile_id"] = self._portable_profile_reference(
                profile.get("profile_id"),
                dependencies,
                "chat",
                allow_auto=False,
            )

    def _import_vision_provenance(
        self,
        provenance: dict[str, Any],
        dependencies: ImportedDependencies | None,
    ) -> None:
        context = provenance.get("context")
        vision = context.get("vision") if isinstance(context, dict) else None
        if not isinstance(vision, dict):
            return
        vision["profile_id"] = self._import_profile_reference(
            None,
            dependencies,
            vision.get("profile_id"),
            "chat",
            allow_auto=False,
        )
        profile = vision.get("profile")
        if isinstance(profile, dict):
            profile.pop("install_id", None)
            profile["profile_id"] = self._import_profile_reference(
                None,
                dependencies,
                profile.get("profile_id"),
                "chat",
                allow_auto=False,
            )

    @classmethod
    def _remap_artifact_references(
        cls,
        value: dict[str, Any],
        mappings: dict[str, str],
        declared_ids: set[str],
        *,
        strict: bool,
    ) -> tuple[dict[str, Any], int]:
        missing = 0
        singular_keys = {
            "artifact_id",
            "browser_proxy_artifact_id",
            "poster_artifact_id",
            "poster_for",
            "proxy_for",
        }
        plural_keys = {"artifact_ids", "input_artifact_ids"}

        def remap(current: Any) -> Any:
            nonlocal missing
            if isinstance(current, dict):
                result: dict[str, Any] = {}
                for key, child in current.items():
                    if key in singular_keys and isinstance(child, str):
                        if child not in declared_ids:
                            if strict:
                                raise ValueError(
                                    "project provenance references an undeclared artifact"
                                )
                            missing += 1
                            result[key] = None
                        else:
                            mapped = mappings.get(child)
                            if mapped is None:
                                missing += 1
                            result[key] = mapped
                    elif key in plural_keys and isinstance(child, list):
                        mapped_ids: list[str] = []
                        for source_id in child:
                            if not isinstance(source_id, str) or source_id not in declared_ids:
                                if strict:
                                    raise ValueError(
                                        "project provenance references an undeclared artifact"
                                    )
                                missing += 1
                                continue
                            mapped = mappings.get(source_id)
                            if mapped is None:
                                missing += 1
                            else:
                                mapped_ids.append(mapped)
                        result[key] = mapped_ids
                    else:
                        result[key] = remap(child)
                return result
            if isinstance(current, list):
                return [remap(child) for child in current]
            return current

        return remap(value), missing

    def import_archive(
        self,
        session: Session,
        source: IO[bytes],
        *,
        known_fields: Mapping[str, list[SettingField]] | None = None,
    ) -> Project:
        """Import a project archive.

        `known_fields` is the live engine schema per role, resolved by the
        caller because only the request layer can await it. When a role is
        absent - an engine that is not configured, or whose schema cannot be
        read right now - that role's settings are imported unvalidated rather
        than dropped: refusing to import because an engine is down would be
        worse than importing something the user can correct.
        """
        self._known_fields = dict(known_fields or {})
        source.seek(0)
        cas_transaction = _ImportCasTransaction(session, self.artifacts.root)
        try:
            with zipfile.ZipFile(source) as archive:
                infos = self._validate_archive(archive)
                manifest_info = infos.get("manifest.json")
                if not manifest_info or manifest_info.file_size > 50 * 1024 * 1024:
                    raise ValueError("project archive has no valid manifest")
                manifest = json.loads(
                    archive.read(manifest_info),
                    parse_constant=self._reject_json_constant,
                )
                self._validate_json_tree(manifest)
                self._validate_manifest(manifest)
                dependency_model = (
                    parse_dependency_manifest(manifest.get("dependencies"))
                    if manifest["version"] >= 3
                    else None
                )
                dependency_index = (
                    dependency_source_index(dependency_model) if dependency_model else None
                )
                self._validate_record_graph(manifest, dependency_index)
                auxiliary_mappings = self._resolve_auxiliary_requirements(session, manifest)
                self._remap_auxiliary_asset_references(manifest, auxiliary_mappings)
                if manifest["version"] >= 3:
                    self._validate_v3_archive_entries(infos, manifest)
                artifact_map = self._import_artifacts(
                    session,
                    archive,
                    infos,
                    manifest,
                    cas_transaction,
                )
                imported_dependencies = (
                    install_dependency_manifest(session, dependency_model)
                    if dependency_model
                    else None
                )
                return self._import_records(
                    session,
                    manifest,
                    artifact_map,
                    imported_dependencies,
                )
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            RecursionError,
            zipfile.BadZipFile,
            zipfile.LargeZipFile,
        ) as exc:
            cas_transaction.rollback()
            raise ValueError("invalid LM Atelier project archive") from exc
        except BaseException:
            cas_transaction.rollback()
            raise

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
            file_type = stat.S_IFMT(mode)
            if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError("project archive cannot contain special files")
            if info.flag_bits & 0x1:
                raise ValueError("project archive cannot contain encrypted entries")
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError("project archive uses an unsupported compression method")
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
        version = manifest.get("version")
        if (
            manifest.get("format") != "local-lm-project"
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version not in {1, 2, 3, 4, 5, 6}
        ):
            raise ValueError("unsupported project archive format")
        if not isinstance(manifest.get("project"), dict):
            raise ValueError("project manifest is missing project metadata")
        for key, maximum in (("chats", 10_000), ("runs", 100_000), ("artifacts", 100_000)):
            value = manifest.get(key)
            if not isinstance(value, list) or len(value) > maximum:
                raise ValueError(f"project manifest has invalid {key}")
        media_included = manifest.get("media_included")
        if not isinstance(media_included, bool):
            raise ValueError("project manifest has an invalid media inclusion flag")
        if version >= 3 and not isinstance(manifest.get("dependencies"), dict):
            raise ValueError("project manifest is missing portable dependencies")
        if version >= 6:
            ProjectExporter._validate_auxiliary_requirements(manifest.get("auxiliary_requirements"))

    @staticmethod
    def _reject_json_constant(value: str) -> None:
        raise ValueError(f"project manifest contains an invalid numeric value: {value}")

    @staticmethod
    def _validate_json_tree(value: object) -> None:
        stack: list[tuple[object, int]] = [(value, 0)]
        nodes = 0
        while stack:
            current, depth = stack.pop()
            nodes += 1
            if nodes > 1_000_000:
                raise ValueError("project manifest contains too many JSON values")
            if depth > 64:
                raise ValueError("project manifest JSON is nested too deeply")
            if isinstance(current, dict):
                if len(current) > 100_000:
                    raise ValueError("project manifest contains an oversized object")
                for key, child in current.items():
                    if not isinstance(key, str) or len(key) > 1_000:
                        raise ValueError("project manifest contains an invalid object key")
                    stack.append((child, depth + 1))
            elif isinstance(current, list):
                if len(current) > 100_000:
                    raise ValueError("project manifest contains an oversized array")
                stack.extend((child, depth + 1) for child in current)
            elif isinstance(current, str):
                if len(current) > 10_000_000:
                    raise ValueError("project manifest contains an oversized string")
            elif isinstance(current, float):
                if not math.isfinite(current):
                    raise ValueError("project manifest contains a non-finite number")
            elif current is None or isinstance(current, (bool, int)):
                continue
            else:
                raise ValueError("project manifest contains a non-JSON value")

    def _validate_record_graph(
        self,
        manifest: dict[str, Any],
        dependencies: DependencySourceIndex | None,
    ) -> None:
        version = manifest["version"]
        artifact_ids = self._validate_artifact_records(manifest["artifacts"], version)
        project = manifest["project"]
        project_id = self._text(project.get("id"), "project id", 40)
        self._text(project.get("name"), "project name", 200)
        self._optional_text(project.get("description"), 10_000)
        self._optional_text(project.get("instructions"), 100_000)
        self._optional_bool(project, "archived")
        self._validate_generation_defaults_record(project, dependencies)
        self._validate_revision_source(
            project.get("image_workflow_revision_id"),
            dependencies,
            {Operation.TEXT_TO_IMAGE.value, Operation.IMAGE_TO_IMAGE.value},
        )
        self._validate_revision_source(
            project.get("video_workflow_revision_id"),
            dependencies,
            {Operation.TEXT_TO_VIDEO.value, Operation.IMAGE_TO_VIDEO.value},
        )

        chat_ids: set[str] = set()
        message_chats: dict[str, str] = {}
        message_roles: dict[str, str] = {}
        part_ids: set[str] = set()
        parent_references: list[tuple[str, str, str]] = []
        active_heads: list[tuple[str, str]] = []
        active_revision_references: list[tuple[str, str]] = []
        revision_run_references: list[tuple[str, str, str]] = []
        revision_runs_by_message: dict[str, set[str]] = {}
        revision_ids: set[str] = set()
        revision_messages: dict[str, str] = {}
        generation_metadata_runs: list[tuple[str, object]] = []
        for chat_data in manifest["chats"]:
            if not isinstance(chat_data, dict):
                raise ValueError("project manifest has an invalid chat")
            chat_id = self._text(chat_data.get("id"), "chat id", 40)
            if chat_id in chat_ids:
                raise ValueError("project manifest contains duplicate chat ids")
            chat_ids.add(chat_id)
            source_project_id = chat_data.get("project_id")
            if source_project_id is not None and source_project_id != project_id:
                raise ValueError("project manifest chat references a different project")
            self._text(chat_data.get("title"), "chat title", 240)
            self._optional_bool(chat_data, "archived")
            self._optional_bool(chat_data, "confirm_uncertain_media")
            try:
                RoutingMode(str(chat_data.get("routing_mode", "auto")))
            except ValueError as exc:
                raise ValueError("project manifest has an invalid chat routing mode") from exc
            self._validate_profile_source(
                chat_data.get("active_chat_profile_id"),
                dependencies,
                "chat",
                allow_auto=True,
            )
            if version >= 5:
                self._validate_profile_source(
                    chat_data.get("active_vision_profile_id"),
                    dependencies,
                    "chat",
                    allow_auto=True,
                )
                self._validate_vision_settings(chat_data.get("vision_settings_json"))
            self._validate_profile_source(
                chat_data.get("active_image_profile_id"),
                dependencies,
                "image",
                allow_auto=True,
            )
            self._validate_profile_source(
                chat_data.get("active_video_profile_id"),
                dependencies,
                "video",
                allow_auto=True,
            )
            self._validate_generation_defaults_record(chat_data, dependencies)
            messages = chat_data.get("messages")
            if not isinstance(messages, list) or len(messages) > 100_000:
                raise ValueError("project manifest has invalid messages")
            for message_data in messages:
                if not isinstance(message_data, dict):
                    raise ValueError("project manifest has an invalid message")
                message_id = self._text(message_data.get("id"), "message id", 40)
                if message_id in message_chats:
                    raise ValueError("project manifest contains duplicate message ids")
                source_chat_id = message_data.get("chat_id")
                if source_chat_id is not None and source_chat_id != chat_id:
                    raise ValueError("project manifest message references a different chat")
                try:
                    role = MessageRole(str(message_data.get("role"))).value
                    _message_status = MessageStatus(str(message_data.get("status")))
                except ValueError as exc:
                    raise ValueError("project manifest has invalid message state") from exc
                if version >= 4 and not isinstance(message_data.get("transcript_visible"), bool):
                    raise ValueError("project manifest has invalid transcript visibility")
                message_chats[message_id] = chat_id
                message_roles[message_id] = role
                parent_id = message_data.get("parent_id")
                if parent_id is not None:
                    parent_references.append(
                        (chat_id, message_id, self._text(parent_id, "parent message id", 40))
                    )
                parts = message_data.get("parts")
                if not isinstance(parts, list) or len(parts) > 10_000:
                    raise ValueError("project manifest has invalid message parts")
                for expected_position, part_data in enumerate(parts):
                    if not isinstance(part_data, dict):
                        raise ValueError("project manifest has an invalid message part")
                    part_id = part_data.get("id")
                    if part_id is not None:
                        part_id = self._text(part_id, "message part id", 40)
                        if part_id in part_ids:
                            raise ValueError("project manifest contains duplicate message part ids")
                        part_ids.add(part_id)
                    position = part_data.get("position")
                    if position is not None and (
                        isinstance(position, bool)
                        or not isinstance(position, int)
                        or position != expected_position
                    ):
                        raise ValueError("project manifest has invalid message part positions")
                    try:
                        _part_type = PartType(str(part_data.get("type")))
                    except ValueError as exc:
                        raise ValueError(
                            "project manifest has an invalid message part type"
                        ) from exc
                    self._optional_text(part_data.get("text"), 10_000_000)
                    metadata = part_data.get("metadata_json")
                    if not isinstance(metadata, dict):
                        raise ValueError("project manifest has invalid message part metadata")
                    if _part_type == PartType.GENERATION_METADATA:
                        generation_metadata_runs.append((message_id, metadata.get("run_id")))
                    artifact_id = part_data.get("artifact_id")
                    if artifact_id is not None:
                        artifact_id = self._text(artifact_id, "message artifact id", 80)
                        if artifact_id not in artifact_ids:
                            raise ValueError(
                                "project manifest message references an undeclared artifact"
                            )
                if version >= 4:
                    revisions = message_data.get("response_revisions")
                    if not isinstance(revisions, list) or len(revisions) > 1_000:
                        raise ValueError("project manifest has invalid response revisions")
                    seen_sequences: set[int] = set()
                    for revision_data in revisions:
                        if not isinstance(revision_data, dict):
                            raise ValueError("project manifest has an invalid response revision")
                        revision_id = self._text(
                            revision_data.get("id"),
                            "response revision id",
                            40,
                        )
                        if revision_id in revision_ids:
                            raise ValueError(
                                "project manifest contains duplicate response revision ids"
                            )
                        revision_ids.add(revision_id)
                        revision_messages[revision_id] = message_id
                        if revision_data.get("message_id") != message_id:
                            raise ValueError(
                                "project response revision references a different message"
                            )
                        sequence = revision_data.get("sequence")
                        if (
                            isinstance(sequence, bool)
                            or not isinstance(sequence, int)
                            or sequence < 1
                            or sequence in seen_sequences
                        ):
                            raise ValueError(
                                "project manifest has an invalid response revision sequence"
                            )
                        seen_sequences.add(sequence)
                        try:
                            MessageStatus(str(revision_data.get("status")))
                        except ValueError as exc:
                            raise ValueError(
                                "project manifest has invalid response revision state"
                            ) from exc
                        revision_run_id = revision_data.get("run_id")
                        if revision_run_id is not None:
                            validated_revision_run_id = self._text(
                                revision_run_id,
                                "response revision run id",
                                40,
                            )
                            revision_run_references.append(
                                (
                                    revision_id,
                                    chat_id,
                                    validated_revision_run_id,
                                )
                            )
                            revision_runs_by_message.setdefault(message_id, set()).add(
                                validated_revision_run_id
                            )
                        revision_parts = revision_data.get("parts")
                        if not isinstance(revision_parts, list) or len(revision_parts) > 10_000:
                            raise ValueError("project manifest has invalid response revision parts")
                        for expected_position, part_data in enumerate(revision_parts):
                            if not isinstance(part_data, dict):
                                raise ValueError(
                                    "project manifest has an invalid response revision part"
                                )
                            position = part_data.get("position")
                            if position is not None and (
                                isinstance(position, bool)
                                or not isinstance(position, int)
                                or position != expected_position
                            ):
                                raise ValueError(
                                    "project manifest has invalid response revision positions"
                                )
                            try:
                                PartType(str(part_data.get("type")))
                            except ValueError as exc:
                                raise ValueError(
                                    "project manifest has an invalid response revision part type"
                                ) from exc
                            self._optional_text(part_data.get("text"), 10_000_000)
                            if not isinstance(part_data.get("metadata_json"), dict):
                                raise ValueError(
                                    "project response revision has invalid part metadata"
                                )
                            artifact_id = part_data.get("artifact_id")
                            if (
                                artifact_id is not None
                                and self._text(
                                    artifact_id,
                                    "response revision artifact id",
                                    80,
                                )
                                not in artifact_ids
                            ):
                                raise ValueError(
                                    "project response revision references an undeclared artifact"
                                )
                    active_revision = message_data.get("active_response_revision_id")
                    if active_revision is not None:
                        active_revision_references.append(
                            (
                                message_id,
                                self._text(
                                    active_revision,
                                    "active response revision id",
                                    40,
                                ),
                            )
                        )
            active_head = chat_data.get("active_head_message_id")
            if active_head is not None:
                active_heads.append(
                    (chat_id, self._text(active_head, "active head message id", 40))
                )

        for chat_id, _message_id, parent_id in parent_references:
            if message_chats.get(parent_id) != chat_id:
                raise ValueError("project manifest has an invalid message parent")
        for chat_id, active_head in active_heads:
            if message_chats.get(active_head) != chat_id:
                raise ValueError("project manifest has an invalid active chat head")
        for message_id, active_revision_id in active_revision_references:
            if revision_messages.get(active_revision_id) != message_id:
                raise ValueError("project manifest has an invalid active response revision")

        run_ids: set[str] = set()
        run_chats: dict[str, str] = {}
        assistant_message_ids: set[str] = set()
        runs_by_assistant_message: dict[str, str] = {}
        for run_data in manifest["runs"]:
            if not isinstance(run_data, dict):
                raise ValueError("project manifest has an invalid run")
            run_id = self._text(run_data.get("id"), "run id", 40)
            if run_id in run_ids:
                raise ValueError("project manifest contains duplicate run ids")
            run_ids.add(run_id)
            chat_id = self._text(run_data.get("chat_id"), "run chat id", 40)
            run_chats[run_id] = chat_id
            user_message_id = self._text(run_data.get("user_message_id"), "run user message id", 40)
            assistant_message_id = self._text(
                run_data.get("assistant_message_id"), "run assistant message id", 40
            )
            if (
                chat_id not in chat_ids
                or message_chats.get(user_message_id) != chat_id
                or message_chats.get(assistant_message_id) != chat_id
                or message_roles.get(user_message_id) != MessageRole.USER.value
                or message_roles.get(assistant_message_id) != MessageRole.ASSISTANT.value
            ):
                raise ValueError("project run references an incompatible chat or message")
            if assistant_message_id in assistant_message_ids:
                raise ValueError("project manifest has duplicate assistant run targets")
            assistant_message_ids.add(assistant_message_id)
            runs_by_assistant_message[assistant_message_id] = run_id
            try:
                operation = Operation(str(run_data.get("operation")))
                RunStatus(str(run_data.get("status", "failed")))
            except ValueError as exc:
                raise ValueError("project manifest has invalid run state") from exc
            self._optional_text(run_data.get("idempotency_key"), 200)
            self._optional_text(run_data.get("standalone_prompt"), 10_000_000)
            self._optional_text(run_data.get("error"), 1_000_000)
            if not isinstance(run_data.get("settings_json"), dict) or not isinstance(
                run_data.get("provenance_json"), dict
            ):
                raise ValueError("project manifest has invalid run settings or provenance")
            if version >= 3:
                self._validate_portable_provenance(
                    run_data["provenance_json"],
                    operation,
                    dependencies,
                    artifact_ids,
                )
            self._validate_profile_source(
                run_data.get("profile_id"),
                dependencies,
                self._role_for_operation(operation),
                allow_auto=False,
            )
            if version >= 5:
                self._validate_profile_source(
                    run_data.get("vision_profile_id"),
                    dependencies,
                    "chat",
                    allow_auto=False,
                )
            self._validate_revision_source(
                run_data.get("workflow_revision_id"),
                dependencies,
                {operation.value},
            )
        if version >= 3:
            for message_id, supplied_run_id in generation_metadata_runs:
                if supplied_run_id is None:
                    continue
                if version >= 4:
                    allowed_run_ids = set(revision_runs_by_message.get(message_id, set()))
                    direct_run_id = runs_by_assistant_message.get(message_id)
                    if direct_run_id:
                        allowed_run_ids.add(direct_run_id)
                    if supplied_run_id not in allowed_run_ids:
                        raise ValueError(
                            "project generation metadata references an incompatible run"
                        )
                elif supplied_run_id != runs_by_assistant_message.get(message_id):
                    raise ValueError("project generation metadata references an incompatible run")
        if version >= 4:
            for _revision_id, chat_id, run_id in revision_run_references:
                if run_chats.get(run_id) != chat_id:
                    raise ValueError("project response revision references an incompatible run")

    def _validate_portable_provenance(
        self,
        value: dict[str, Any],
        operation: Operation,
        dependencies: DependencySourceIndex | None,
        artifact_ids: set[str],
    ) -> None:
        role = self._role_for_operation(operation)
        self._remap_artifact_references(
            value,
            {artifact_id: artifact_id for artifact_id in artifact_ids},
            artifact_ids,
            strict=True,
        )

        model = value.get("model")
        if isinstance(model, dict):
            self._validate_profile_source(
                model.get("profile_id"),
                dependencies,
                role,
                allow_auto=False,
            )
        selection = value.get("model_selection")
        if isinstance(selection, dict):
            self._validate_profile_source(
                selection.get("profile_id"),
                dependencies,
                role,
                allow_auto=False,
            )
        self._validate_vision_provenance(value, dependencies)
        preset = value.get("preset")
        if isinstance(preset, dict):
            self._validate_preset_source(preset.get("id"), dependencies, role)
        layers = value.get("preset_layers")
        if isinstance(layers, list):
            for layer in layers:
                if isinstance(layer, dict):
                    self._validate_preset_source(layer.get("id"), dependencies, role)

        workflow = value.get("workflow")
        if not isinstance(workflow, dict):
            return
        revision_id = workflow.get("revision_id")
        self._validate_revision_source(
            revision_id,
            dependencies,
            {operation.value},
        )
        supplied_workflow_id = workflow.get("definition_id")
        expected_workflow_id = (
            dependencies.revision_workflow_ids.get(revision_id)
            if dependencies and isinstance(revision_id, str)
            else None
        )
        if supplied_workflow_id != expected_workflow_id:
            raise ValueError("project provenance has mismatched workflow identifiers")

    def _validate_vision_provenance(
        self,
        provenance: dict[str, Any],
        dependencies: DependencySourceIndex | None,
    ) -> None:
        context = provenance.get("context")
        vision = context.get("vision") if isinstance(context, dict) else None
        if not isinstance(vision, dict):
            return
        self._validate_profile_source(
            vision.get("profile_id"),
            dependencies,
            "chat",
            allow_auto=False,
        )
        profile = vision.get("profile")
        if isinstance(profile, dict):
            self._validate_profile_source(
                profile.get("profile_id"),
                dependencies,
                "chat",
                allow_auto=False,
            )

    def _validate_artifact_records(self, records: list[Any], version: int) -> set[str]:
        artifact_ids: set[str] = set()
        digests: set[str] = set()
        archive_paths: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("project manifest has an invalid artifact record")
            artifact_id = self._text(record.get("id"), "artifact id", 80)
            if artifact_id in artifact_ids:
                raise ValueError("project manifest contains duplicate artifact ids")
            artifact_ids.add(artifact_id)
            digest = self._text(record.get("sha256"), "artifact checksum", 64)
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("project manifest has an invalid artifact checksum")
            if digest in digests:
                raise ValueError("project manifest contains duplicate artifact checksums")
            digests.add(digest)
            archive_path = self._text(record.get("archive_path"), "artifact path", 1_000)
            path = PurePosixPath(archive_path)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in archive_path
                or len(path.parts) != 3
                or path.parts[:2] != ("artifacts", digest)
                or not path.name
            ):
                raise ValueError("project manifest has an invalid artifact path")
            if archive_path in archive_paths:
                raise ValueError("project manifest contains duplicate artifact paths")
            archive_paths.add(archive_path)
            size = record.get("size_bytes")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or size > self.settings.max_project_import_bytes
            ):
                raise ValueError("project manifest has an invalid artifact size")
            self._text(record.get("media_type"), "media type", 120)
            self._optional_text(record.get("original_name"), 500)
            if not isinstance(record.get("metadata"), dict):
                raise ValueError("project manifest has invalid artifact metadata")
            try:
                ArtifactKind(str(record.get("kind")))
            except ValueError as exc:
                if version >= 3:
                    raise ValueError("project manifest has an invalid artifact kind") from exc
        return artifact_ids

    @staticmethod
    def _optional_bool(record: dict[str, Any], key: str) -> None:
        if key in record and not isinstance(record[key], bool):
            raise ValueError(f"project manifest has an invalid {key.replace('_', ' ')}")

    @staticmethod
    def _validate_profile_source(
        value: object,
        dependencies: DependencySourceIndex | None,
        expected_role: str,
        *,
        allow_auto: bool,
    ) -> None:
        if value is None or (allow_auto and value == AUTO_PROFILE_ID):
            return
        if not isinstance(value, str) or not value or len(value) > 80:
            raise ValueError("project manifest has an invalid profile reference")
        if dependencies and dependencies.profile_roles.get(value) != expected_role:
            raise ValueError("project manifest references a profile with an incompatible role")

    @staticmethod
    def _validate_revision_source(
        value: object,
        dependencies: DependencySourceIndex | None,
        operations: set[str],
    ) -> None:
        if value is None:
            return
        if not isinstance(value, str) or not value or len(value) > 80:
            raise ValueError("project manifest has an invalid workflow revision reference")
        if dependencies and dependencies.revision_operations.get(value) not in operations:
            raise ValueError(
                "project manifest references a workflow revision with an incompatible operation"
            )

    @staticmethod
    def _validate_preset_source(
        value: object,
        dependencies: DependencySourceIndex | None,
        expected_role: str,
    ) -> None:
        if value is None:
            return
        if not isinstance(value, str) or not value or len(value) > 80:
            raise ValueError("project manifest has an invalid generation preset reference")
        if dependencies and dependencies.preset_roles.get(value) != expected_role:
            raise ValueError(
                "project manifest references a generation preset with an incompatible role"
            )

    @staticmethod
    def _validate_generation_defaults_record(
        record: dict[str, Any],
        dependencies: DependencySourceIndex | None,
    ) -> None:
        settings_by_role = record.get("generation_settings_json", {})
        if not isinstance(settings_by_role, dict):
            raise ValueError("project manifest has invalid generation settings")
        for role, settings in settings_by_role.items():
            if role not in {"chat", "image", "video"} or not isinstance(settings, dict):
                raise ValueError("project manifest has invalid generation settings")
            if len(settings) > 256 or any(
                not isinstance(key, str) or not key or len(key) > 200 for key in settings
            ):
                raise ValueError("project manifest has oversized generation settings")
        bindings = record.get("generation_preset_ids_json", {})
        if not isinstance(bindings, dict):
            raise ValueError("project manifest has invalid generation preset bindings")
        for role, preset_id in bindings.items():
            if role not in {"chat", "image", "video"}:
                raise ValueError("project manifest has invalid generation preset bindings")
            if preset_id is None:
                continue
            if not isinstance(preset_id, str) or not preset_id or len(preset_id) > 80:
                raise ValueError("project manifest has an invalid generation preset reference")
            if dependencies and dependencies.preset_roles.get(preset_id) != role:
                raise ValueError(
                    "project manifest references a generation preset with an incompatible role"
                )

    @staticmethod
    def _validate_vision_settings(value: object) -> None:
        # Unknown keys are still refused - an archive must not smuggle settings
        # this build cannot describe - but the set of known keys is read from the
        # model. A hand-written copy of it made every new vision setting reject
        # archives this same build had just written.
        if not isinstance(value, dict) or set(value) - set(VisionSettings.model_fields):
            raise ValueError("project manifest has invalid vision settings")
        try:
            VisionSettings.model_validate(value, strict=True)
        except ValidationError as exc:
            raise ValueError("project manifest has invalid vision settings") from exc

    @staticmethod
    def _validate_v3_archive_entries(
        infos: dict[str, zipfile.ZipInfo],
        manifest: dict[str, Any],
    ) -> None:
        declared = {str(record["archive_path"]) for record in manifest["artifacts"]}
        allowed = {"manifest.json"}
        if manifest["media_included"]:
            allowed.update(declared)
        actual = set(infos)
        if actual - allowed:
            raise ValueError("project archive contains files that are not declared in its manifest")
        if manifest["media_included"] and declared - actual:
            raise ValueError("project archive is missing media declared in its manifest")

    def _import_artifacts(
        self,
        session: Session,
        archive: zipfile.ZipFile,
        infos: dict[str, zipfile.ZipInfo],
        manifest: dict[str, Any],
        cas_transaction: _ImportCasTransaction,
    ) -> dict[str, str]:
        imported: dict[str, str] = {}
        seen: set[str] = set()
        declared_artifact_ids = {
            str(record["id"])
            for record in manifest["artifacts"]
            if isinstance(record, dict) and isinstance(record.get("id"), str)
        }
        planned_mappings = {
            str(record["id"]): f"sha256:{record['sha256']}"
            for record in manifest["artifacts"]
            if isinstance(record, dict)
            and isinstance(record.get("id"), str)
            and isinstance(record.get("sha256"), str)
            and isinstance(record.get("archive_path"), str)
            and record["archive_path"] in infos
        }
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
            destination = (self.artifacts.root / digest[:2] / digest[2:4] / digest).resolve()
            if self.artifacts.root not in destination.parents:
                raise ValueError("project artifact destination escapes the content store")
            existed_before = destination.is_file()
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
                metadata = redact_local_paths(record.get("metadata"))
                metadata, _missing_artifact_count = self._remap_artifact_references(
                    metadata if isinstance(metadata, dict) else {},
                    planned_mappings,
                    declared_artifact_ids,
                    strict=manifest["version"] >= 3,
                )
                try:
                    artifact = self.artifacts.ingest_stream(
                        session,
                        staged,
                        kind=kind,
                        media_type=self._text(record.get("media_type"), "media type", 120),
                        original_name=self._optional_text(
                            redact_local_paths(record.get("original_name")),
                            500,
                        ),
                        metadata=metadata if isinstance(metadata, dict) else {},
                    )
                finally:
                    cas_transaction.track_created(
                        destination,
                        existed_before=existed_before,
                    )
                if artifact.id != planned_mappings[old_id]:
                    raise ValueError("project artifact identity does not match its checksum")
                if not destination.is_file() or destination.stat().st_size != expected_size:
                    raise ValueError("project content store has a conflicting artifact")
                if existed_before:
                    stored_checksum = hashlib.sha256()
                    with destination.open("rb") as stored:
                        while stored_chunk := stored.read(1024 * 1024):
                            stored_checksum.update(stored_chunk)
                    if stored_checksum.hexdigest() != digest:
                        raise ValueError("project content store has a corrupt artifact")
            imported[old_id] = artifact.id
        return imported

    def _import_records(
        self,
        session: Session,
        manifest: dict[str, Any],
        artifacts: dict[str, str],
        dependencies: ImportedDependencies | None,
    ) -> Project:
        strict_portability = manifest["version"] >= 3
        declared_artifact_ids = {
            str(record["id"])
            for record in manifest["artifacts"]
            if isinstance(record, dict) and isinstance(record.get("id"), str)
        }
        source_runs = {
            str(record["id"]): record
            for record in manifest["runs"]
            if isinstance(record, dict) and isinstance(record.get("id"), str)
        }
        source_runs_by_assistant = {
            str(record["assistant_message_id"]): record
            for record in source_runs.values()
            if isinstance(record.get("assistant_message_id"), str)
        }
        project_data = manifest["project"]
        project = Project(
            name=self._text(project_data.get("name"), "project name", 200),
            description=self._optional_text(project_data.get("description"), 10_000) or "",
            instructions=self._optional_text(project_data.get("instructions"), 100_000) or "",
            archived=bool(project_data.get("archived", False)),
            image_workflow_revision_id=self._import_workflow_revision(
                session,
                dependencies,
                project_data.get("image_workflow_revision_id"),
                {Operation.TEXT_TO_IMAGE.value, Operation.IMAGE_TO_IMAGE.value},
            ),
            video_workflow_revision_id=self._import_workflow_revision(
                session,
                dependencies,
                project_data.get("video_workflow_revision_id"),
                {Operation.TEXT_TO_VIDEO.value, Operation.IMAGE_TO_VIDEO.value},
            ),
            generation_settings_json=self._generation_settings(
                project_data.get("generation_settings_json")
            ),
            generation_preset_ids_json=self._import_preset_bindings(
                session,
                dependencies,
                project_data.get("generation_preset_ids_json"),
            ),
        )
        session.add(project)
        session.flush()
        chat_map: dict[str, Chat] = {}
        message_map: dict[str, Message] = {}
        chat_records: dict[str, dict[str, Any]] = {}
        generation_metadata_parts: list[tuple[MessagePart, str]] = []
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
                active_chat_profile_id=self._import_profile_reference(
                    session,
                    dependencies,
                    chat_data.get("active_chat_profile_id"),
                    "chat",
                ),
                active_vision_profile_id=self._import_profile_reference(
                    session,
                    dependencies,
                    chat_data.get("active_vision_profile_id")
                    if manifest["version"] >= 5
                    else AUTO_PROFILE_ID,
                    "chat",
                ),
                active_image_profile_id=self._import_profile_reference(
                    session,
                    dependencies,
                    chat_data.get("active_image_profile_id"),
                    "image",
                ),
                active_video_profile_id=self._import_profile_reference(
                    session,
                    dependencies,
                    chat_data.get("active_video_profile_id"),
                    "video",
                ),
                generation_settings_json=self._generation_settings(
                    chat_data.get("generation_settings_json")
                ),
                generation_preset_ids_json=self._import_preset_bindings(
                    session,
                    dependencies,
                    chat_data.get("generation_preset_ids_json"),
                ),
                vision_settings_json=self._vision_settings(
                    chat_data.get("vision_settings_json") if manifest["version"] >= 5 else None
                ),
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
                    transcript_visible=bool(message_data.get("transcript_visible", True)),
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
                    metadata = redact_local_paths(metadata) if isinstance(metadata, dict) else {}
                    metadata, _missing_artifact_count = self._remap_artifact_references(
                        metadata,
                        artifacts,
                        declared_artifact_ids,
                        strict=strict_portability,
                    )
                    artifact_id = (
                        artifacts.get(old_artifact_id) if isinstance(old_artifact_id, str) else None
                    )
                    if old_artifact_id and not artifact_id:
                        metadata["missing_import_artifact_id"] = old_artifact_id
                    part_type = PartType(str(part_data.get("type")))
                    part = MessagePart(
                        position=len(message.parts),
                        type=part_type.value,
                        text=self._optional_text(part_data.get("text"), 10_000_000),
                        artifact_id=artifact_id,
                        metadata_json=metadata,
                    )
                    message.parts.append(part)
                    if part_type == PartType.GENERATION_METADATA:
                        source_run = source_runs_by_assistant.get(old_message_id)
                        supplied_run_id = metadata.get("run_id")
                        if (
                            manifest["version"] >= 4
                            and isinstance(supplied_run_id, str)
                            and supplied_run_id in source_runs
                        ):
                            generation_metadata_parts.append((part, supplied_run_id))
                        elif source_run:
                            source_run_id = str(source_run["id"])
                            if (
                                strict_portability
                                and manifest["version"] < 4
                                and supplied_run_id is not None
                                and supplied_run_id != source_run_id
                            ):
                                raise ValueError(
                                    "project generation metadata references a different run"
                                )
                            generation_metadata_parts.append((part, source_run_id))
                        else:
                            metadata.pop("run_id", None)
                            metadata.pop("provenance", None)
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
        imported_runs: dict[str, Run] = {}
        for run_data in manifest["runs"]:
            if not isinstance(run_data, dict):
                raise ValueError("project manifest has an invalid run")
            imported_chat = chat_map.get(str(run_data.get("chat_id")))
            user_message = message_map.get(str(run_data.get("user_message_id")))
            assistant_message = message_map.get(str(run_data.get("assistant_message_id")))
            if not imported_chat or not user_message or not assistant_message:
                raise ValueError("project run references a missing chat or message")
            operation = Operation(str(run_data.get("operation")))
            status = RunStatus(str(run_data.get("status", "failed")))
            interrupted = status not in {
                RunStatus.COMPLETE,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }
            provenance = run_data.get("provenance_json")
            provenance = self._import_provenance(
                provenance,
                operation,
                dependencies,
                artifacts,
                declared_artifact_ids,
                strict=strict_portability,
            )
            provenance["imported_from_run_id"] = run_data.get("id")
            imported_run = Run(
                chat_id=imported_chat.id,
                user_message_id=user_message.id,
                assistant_message_id=assistant_message.id,
                operation=operation.value,
                status=RunStatus.FAILED.value if interrupted else status.value,
                standalone_prompt=self._optional_text(run_data.get("standalone_prompt"), 10_000_000)
                or "",
                profile_id=self._import_profile_reference(
                    session,
                    dependencies,
                    run_data.get("profile_id"),
                    self._role_for_operation(operation),
                    allow_auto=False,
                ),
                vision_profile_id=self._import_profile_reference(
                    session,
                    dependencies,
                    run_data.get("vision_profile_id") if manifest["version"] >= 5 else None,
                    "chat",
                    allow_auto=False,
                ),
                workflow_revision_id=self._import_workflow_revision(
                    session,
                    dependencies,
                    run_data.get("workflow_revision_id"),
                    {operation.value},
                ),
                settings_json=redact_local_paths(run_data.get("settings_json"))
                if isinstance(run_data.get("settings_json"), dict)
                else {},
                provenance_json=provenance,
                error="Imported while generation was incomplete."
                if interrupted
                else self._optional_text(
                    redact_local_paths(run_data.get("error")),
                    1_000_000,
                ),
            )
            session.add(imported_run)
            session.flush()
            imported_runs[str(run_data["id"])] = imported_run
        for chat_data in manifest["chats"]:
            if not isinstance(chat_data, dict):
                continue
            for message_data in chat_data.get("messages", []):
                if not isinstance(message_data, dict):
                    continue
                old_message_id = str(message_data.get("id"))
                imported_message = message_map.get(old_message_id)
                if not imported_message or imported_message.role != MessageRole.ASSISTANT.value:
                    continue
                if manifest["version"] >= 4:
                    revision_map: dict[str, ResponseRevision] = {}
                    revisions = message_data.get("response_revisions")
                    if not isinstance(revisions, list):
                        revisions = []
                    for revision_data in revisions:
                        if not isinstance(revision_data, dict):
                            continue
                        revision_source_run_id = revision_data.get("run_id")
                        linked_run = (
                            imported_runs.get(revision_source_run_id)
                            if isinstance(revision_source_run_id, str)
                            else None
                        )
                        source_status = MessageStatus(str(revision_data.get("status")))
                        revision = ResponseRevision(
                            message_id=imported_message.id,
                            run_id=linked_run.id if linked_run else None,
                            sequence=int(revision_data["sequence"]),
                            status=(
                                MessageStatus.FAILED.value
                                if source_status == MessageStatus.PENDING
                                else source_status.value
                            ),
                        )
                        session.add(revision)
                        session.flush()
                        revision_map[str(revision_data["id"])] = revision
                        revision_parts = revision_data.get("parts")
                        if not isinstance(revision_parts, list):
                            continue
                        for part_data in revision_parts:
                            if not isinstance(part_data, dict):
                                continue
                            old_artifact_id = part_data.get("artifact_id")
                            metadata = part_data.get("metadata_json")
                            metadata = (
                                redact_local_paths(metadata) if isinstance(metadata, dict) else {}
                            )
                            metadata, _missing_artifact_count = self._remap_artifact_references(
                                metadata,
                                artifacts,
                                declared_artifact_ids,
                                strict=strict_portability,
                            )
                            metadata_run_id = metadata.get("run_id")
                            resolved_metadata_run = (
                                imported_runs.get(metadata_run_id)
                                if isinstance(metadata_run_id, str)
                                else None
                            )
                            if resolved_metadata_run:
                                metadata["run_id"] = resolved_metadata_run.id
                                if "provenance" in metadata:
                                    metadata["provenance"] = resolved_metadata_run.provenance_json
                            elif metadata_run_id is not None:
                                metadata.pop("run_id", None)
                                metadata.pop("provenance", None)
                            imported_artifact_id = (
                                artifacts.get(old_artifact_id)
                                if isinstance(old_artifact_id, str)
                                else None
                            )
                            if old_artifact_id and not imported_artifact_id:
                                metadata["missing_import_artifact_id"] = old_artifact_id
                            revision.parts.append(
                                ResponseRevisionPart(
                                    position=len(revision.parts),
                                    type=PartType(str(part_data.get("type"))).value,
                                    text=self._optional_text(
                                        part_data.get("text"),
                                        10_000_000,
                                    ),
                                    artifact_id=imported_artifact_id,
                                    metadata_json=metadata,
                                )
                            )
                    active_source_id = message_data.get("active_response_revision_id")
                    active_revision = (
                        revision_map.get(active_source_id)
                        if isinstance(active_source_id, str)
                        else None
                    )
                    if active_revision:
                        imported_message.active_response_revision_id = active_revision.id
                else:
                    source_run = source_runs_by_assistant.get(old_message_id)
                    legacy_imported_run = (
                        imported_runs.get(str(source_run["id"])) if source_run else None
                    )
                    revision = ResponseRevision(
                        message_id=imported_message.id,
                        run_id=legacy_imported_run.id if legacy_imported_run else None,
                        sequence=1,
                        status=imported_message.status,
                        parts=[
                            ResponseRevisionPart(
                                position=part.position,
                                type=part.type,
                                text=part.text,
                                artifact_id=part.artifact_id,
                                metadata_json=dict(part.metadata_json),
                            )
                            for part in imported_message.parts
                        ],
                    )
                    session.add(revision)
                    session.flush()
                    imported_message.active_response_revision_id = revision.id
        for part, source_run_id in generation_metadata_parts:
            resolved_run = imported_runs.get(source_run_id)
            if not resolved_run:
                if strict_portability:
                    raise ValueError("project generation metadata references an unavailable run")
                part.metadata_json.pop("run_id", None)
                part.metadata_json.pop("provenance", None)
                continue
            part.metadata_json = {
                **part.metadata_json,
                "run_id": resolved_run.id,
                "provenance": resolved_run.provenance_json,
            }
        session.flush()
        return project

    @staticmethod
    def _import_workflow_revision(
        session: Session,
        dependencies: ImportedDependencies | None,
        value: object,
        operations: set[str],
    ) -> str | None:
        if dependencies:
            return dependencies.revision(value, operations)
        return None

    @staticmethod
    def _portable_revision_reference(
        value: object,
        dependencies: DependencySourceIndex,
        operations: set[str],
    ) -> str | None:
        if isinstance(value, str) and dependencies.revision_operations.get(value) in operations:
            return value
        return None

    @staticmethod
    def _portable_profile_reference(
        value: object,
        dependencies: DependencySourceIndex,
        expected_role: str,
        *,
        allow_auto: bool = True,
    ) -> str | None:
        if allow_auto and value == AUTO_PROFILE_ID:
            return AUTO_PROFILE_ID
        if isinstance(value, str) and dependencies.profile_roles.get(value) == expected_role:
            return value
        return None

    @staticmethod
    def _portable_preset_reference(
        value: object,
        dependencies: DependencySourceIndex,
        expected_role: str,
    ) -> str | None:
        if isinstance(value, str) and dependencies.preset_roles.get(value) == expected_role:
            return value
        return None

    @staticmethod
    def _import_profile_reference(
        session: Session | None,
        dependencies: ImportedDependencies | None,
        value: object,
        expected_role: str,
        *,
        allow_auto: bool = True,
    ) -> str | None:
        if dependencies:
            return dependencies.profile(value, expected_role, allow_auto=allow_auto)
        if allow_auto and value == AUTO_PROFILE_ID:
            return AUTO_PROFILE_ID
        return None

    @staticmethod
    def _snapshot_generation_defaults(
        session: Session,
        owner: Project | Chat,
        record: dict[str, Any],
        dependencies: DependencySourceIndex,
    ) -> None:
        raw_settings = owner.generation_settings_json
        settings = {
            role: dict(values)
            for role, values in (raw_settings.items() if isinstance(raw_settings, dict) else [])
            if role in {"chat", "image", "video"} and isinstance(values, dict)
        }
        portable_bindings: dict[str, str] = {}
        bindings = owner.generation_preset_ids_json
        for role, preset_id in bindings.items() if isinstance(bindings, dict) else []:
            if role not in {"chat", "image", "video"} or not isinstance(preset_id, str):
                continue
            preset = session.get(GenerationPreset, preset_id)
            if preset and preset.role == role:
                settings[role] = {**preset.settings_json, **settings.get(role, {})}
                if dependencies.preset_roles.get(preset_id) == role:
                    portable_bindings[role] = preset_id
        # The direct settings snapshot preserves effective behavior even when
        # the imported preset is later edited. The remapped binding preserves
        # its reusable project/chat relationship.
        record["generation_settings_json"] = settings
        record["generation_preset_ids_json"] = portable_bindings

    def _generation_settings(self, value: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(value, dict):
            return {}
        redacted = cast(
            dict[str, dict[str, Any]],
            redact_local_paths(
                {
                    role: dict(settings)
                    for role, settings in value.items()
                    if role in {"chat", "image", "video"} and isinstance(settings, dict)
                }
            ),
        )
        return {
            role: self._validated_role_settings(role, settings)
            for role, settings in redacted.items()
        }

    def _validated_role_settings(self, role: str, values: dict[str, Any]) -> dict[str, Any]:
        """Keep the settings this engine actually accepts, and drop the rest.

        Imported settings used to be written verbatim, so a value the REST API
        would reject could arrive through an archive and fail later at
        generation time, where nothing connects the failure to the import.

        Validation is against the live engine schema, not the settings registry.
        An earlier attempt used the registry and silently dropped legitimate
        settings, because a workflow's input schema contributes fields the
        registry does not know about.

        A bad key costs only itself. Rejecting the archive over one stale
        setting would make old exports unimportable after any engine change,
        which is the failure this whole task exists to remove. And a key the role
        schema does not describe is kept, not dropped - see below.
        """
        fields = self._known_fields.get(role)
        if fields is None or not values:
            return values
        described = {field.key: field for field in fields}
        kept: dict[str, Any] = {}
        for key, item in values.items():
            definition = described.get(key)
            if definition is None:
                # Not described by the role schema, which does not mean invalid:
                # a workflow's input schema contributes fields that only appear
                # once that workflow is selected. Validating what can be
                # described must never mean deleting what cannot.
                kept[key] = item
                continue
            try:
                kept.update(validate_settings({key: item}, [definition]))
            except ValueError:
                continue
        return kept

    @staticmethod
    def _vision_settings(value: Any) -> dict[str, Any]:
        candidate = value if isinstance(value, dict) else {}
        try:
            return VisionSettings.model_validate(candidate, strict=True).model_dump(mode="json")
        except ValidationError:
            return VisionSettings().model_dump(mode="json")

    @staticmethod
    def _import_preset_bindings(
        session: Session,
        dependencies: ImportedDependencies | None,
        value: object,
    ) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        if not dependencies:
            return {}
        imported: dict[str, str] = {}
        for role, source_id in value.items():
            if role not in {"chat", "image", "video"}:
                continue
            preset_id = dependencies.preset(source_id, role)
            if preset_id:
                imported[role] = preset_id
        return imported

    @staticmethod
    def _import_preset_reference(
        dependencies: ImportedDependencies | None,
        value: object,
        expected_role: str,
    ) -> str | None:
        if not dependencies:
            return None
        return dependencies.preset(value, expected_role)

    @staticmethod
    def _role_for_operation(operation: Operation) -> str:
        if operation == Operation.TEXT:
            return "chat"
        if operation in {Operation.TEXT_TO_IMAGE, Operation.IMAGE_TO_IMAGE}:
            return "image"
        return "video"

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
