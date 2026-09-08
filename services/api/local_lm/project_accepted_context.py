"""Preserve accepted conversation inputs across portable project identities."""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .accepted_turn_context import AcceptedContext, _digest
from .domain import new_id
from .models import Chat, Message, ResponseRevision, Run, RunContextArtifact, RunContextSnapshot
from .project_dependencies import ImportedDependencies
from .project_portability import has_local_path, redact_local_paths
from .project_work_plans import remap_work_references

if TYPE_CHECKING:
    from .exports import ProjectExporter
    from .project_dependencies import DependencySourceIndex


def removed_at(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("project message has an invalid removal timestamp")
    try:
        result = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("project message has an invalid removal timestamp") from None
    if result.tzinfo is None:
        raise ValueError("project message has an invalid removal timestamp")
    return result


def export_accepted_contexts(
    exporter: ProjectExporter,
    session: Session,
    runs: list[Run],
    run_records: list[dict[str, Any]],
    dependencies: DependencySourceIndex,
) -> list[dict[str, Any]]:
    by_id = {record["id"]: record for record in run_records}
    messages = {
        message.id: message
        for message in session.scalars(
            select(Message).where(Message.chat_id.in_({run.chat_id for run in runs}))
        )
    }
    result = []
    for row in session.scalars(
        select(RunContextSnapshot).where(RunContextSnapshot.run_id.in_(by_id))
    ):
        record = by_id[row.run_id]
        if (
            row.sha256 != record["provenance_json"].get("accepted_context_sha256")
            or _digest(row.payload_json) != row.sha256
        ):
            raise ValueError("project accepted context identity is invalid")
        snapshot = AcceptedContext.model_validate(row.payload_json)
        payload = snapshot.model_dump(mode="json")
        for entry in payload["messages"]:
            source = messages.get(entry["source_message_id"])
            if entry["source_message_id"] is not None and (
                source is None or source.content_removed_at is not None
            ):
                entry["content"] = ""
                entry["content_sha256"] = hashlib.sha256(b"").hexdigest()
                payload["unavailable_reason"] = "removed_context"
        if payload["unavailable_reason"] == "removed_context":
            payload["compiled_prompt"] = None
            payload["standalone_prompt"] = ""
            payload["media_prompt"] = ""
        for field in ("profile", "vision_profile", "verification_profile"):
            profile = payload[field]
            if profile is not None:
                # Install locations and launch bindings belong to the destination.
                profile["install"] = None
                profile["id"] = (
                    exporter._portable_profile_reference(
                        profile["id"],
                        dependencies,
                        profile["role"],
                        allow_auto=False,
                    )
                    or profile["id"]
                )
        workflow = payload["workflow"]
        if workflow is not None:
            workflow["trusted"] = False
        payload["workflow_activation"] = None
        for field in ("preset", "preset_layers", "auxiliary_assets"):
            payload[field] = copy.deepcopy(record["provenance_json"].get(field))
        payload["preset_layers"] = payload["preset_layers"] or []
        payload["auxiliary_assets"] = payload["auxiliary_assets"] or {}
        payload = redact_local_paths(payload)
        # A path scrub may change context text, so bind the portable text itself.
        for entry in payload["messages"]:
            entry["content_sha256"] = hashlib.sha256(entry["content"].encode("utf-8")).hexdigest()
        digest = _digest(payload)
        record["provenance_json"]["accepted_context_sha256"] = digest
        result.append({"payload_json": payload, "sha256": digest})
    return result


def validate_accepted_contexts(manifest: dict[str, Any]) -> list[AcceptedContext]:
    if manifest["version"] < 7:
        return []
    records = manifest.get("accepted_contexts")
    if not isinstance(records, list) or len(records) > 100_000:
        raise ValueError("project manifest has invalid accepted contexts")
    runs = {run["id"]: run for run in manifest["runs"]}
    messages = {
        message["id"]: (chat["id"], message)
        for chat in manifest["chats"]
        for message in chat["messages"]
    }
    revisions = {
        revision["id"]: message["id"]
        for _, message in messages.values()
        for revision in message.get("response_revisions", [])
    }
    steps = {step["id"]: (plan, step) for plan in manifest["work_plans"] for step in plan["steps"]}
    artifact_ids = {artifact["id"] for artifact in manifest["artifacts"]}
    seen: set[str] = set()
    result = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"payload_json", "sha256"}:
            raise ValueError("project manifest has an invalid accepted context")
        payload = record["payload_json"]
        snapshot = AcceptedContext.model_validate(payload)
        run = runs.get(snapshot.run_id)
        if (
            run is None
            or snapshot.run_id in seen
            or snapshot.chat_id != run["chat_id"]
            or snapshot.operation != run["operation"]
            or _digest(payload) != record["sha256"]
            or run["provenance_json"].get("accepted_context_sha256") != record["sha256"]
            or has_local_path(payload)
            or snapshot.workflow_activation is not None
            or any(
                profile is not None and profile.install is not None
                for profile in (
                    snapshot.profile,
                    snapshot.vision_profile,
                    snapshot.verification_profile,
                )
            )
            or (snapshot.workflow is not None and snapshot.workflow.trusted)
        ):
            raise ValueError("project accepted context identity is invalid")
        seen.add(snapshot.run_id)
        if snapshot.unavailable_reason == "removed_context" and (
            snapshot.compiled_prompt is not None
            or snapshot.standalone_prompt
            or snapshot.media_prompt
        ):
            raise ValueError("project accepted context retains removed prompt content")
        for entry in snapshot.messages:
            if hashlib.sha256(entry.content.encode("utf-8")).hexdigest() != entry.content_sha256:
                raise ValueError("project accepted context text identity is invalid")
            if entry.source_message_id is not None:
                source = messages.get(entry.source_message_id)
                if source is not None and source[0] != snapshot.chat_id:
                    raise ValueError("project accepted context references a different chat")
                if (source is None or source[1].get("content_removed_at") is not None) and (
                    snapshot.unavailable_reason != "removed_context" or entry.content
                ):
                    raise ValueError("project accepted context retains removed content")
                if entry.response_revision_id is not None:
                    revision_message = revisions.get(entry.response_revision_id)
                    if revision_message != entry.source_message_id and not (
                        revision_message is None
                        and snapshot.unavailable_reason == "removed_context"
                    ):
                        raise ValueError("project accepted context references a different revision")
        source_run = runs.get(snapshot.source_run_id or "")
        if source_run is not None and (
            source_run["chat_id"] != snapshot.chat_id
            or source_run["user_message_id"] != snapshot.source_message_id
        ):
            raise ValueError("project accepted context has a different edit source")
        if not set(snapshot.artifact_ids) <= artifact_ids:
            raise ValueError("project accepted context references undeclared media")
        for values in (
            snapshot.input_artifact_ids,
            snapshot.visual_artifact_ids,
            snapshot.strict_artifact_ids,
            snapshot.context_artifact_ids or [],
            list(snapshot.visual_posters.values()),
        ):
            if not set(values) <= set(snapshot.artifact_ids):
                raise ValueError("project accepted context has unretained media")
        if not snapshot.visual_posters.keys() <= set(snapshot.visual_artifact_ids):
            raise ValueError("project accepted context has an invalid visual poster")
        for dependency in snapshot.dependencies:
            pair = steps.get(dependency.step_id)
            producer = runs.get(dependency.run_id)
            if pair is None or producer is None:
                if snapshot.unavailable_reason is None:
                    raise ValueError("project accepted dependency is missing")
            elif (
                pair[0]["id"] != dependency.plan_id
                or pair[1]["run_id"] != dependency.run_id
                or producer["chat_id"] != snapshot.chat_id
                or producer["assistant_message_id"] != dependency.message_id
            ):
                raise ValueError("project accepted dependency identity is invalid")
        result.append(snapshot)
    for run_id, run in runs.items():
        if run["provenance_json"].get("accepted_context_sha256") is not None and run_id not in seen:
            raise ValueError("project accepted context is missing")
    return result


def import_accepted_contexts(
    session: Session,
    snapshots: list[AcceptedContext],
    *,
    chats: dict[str, Chat],
    messages: dict[str, Message],
    runs: dict[str, Run],
    revisions: dict[str, str],
    plans: dict[str, str],
    steps: dict[str, str],
    artifacts: dict[str, str],
    dependencies: ImportedDependencies | None,
) -> None:
    run_ids = {key: run.id for key, run in runs.items()}
    message_ids = {key: message.id for key, message in messages.items()}
    missing: dict[str, str] = {}
    for snapshot in snapshots:
        run = runs[snapshot.run_id]
        payload = remap_work_references(
            snapshot.model_dump(mode="json"),
            plans=plans,
            steps=steps,
            runs=run_ids,
            messages=message_ids,
            missing=missing,
        )
        payload["chat_id"] = chats[snapshot.chat_id].id
        for entry, original in zip(payload["messages"], snapshot.messages, strict=True):
            entry["response_revision_id"] = revisions.get(original.response_revision_id or "")
        if set(snapshot.artifact_ids) - artifacts.keys():
            payload["unavailable_reason"] = "imported_media_unavailable"
        for field in (
            "artifact_ids",
            "input_artifact_ids",
            "visual_artifact_ids",
            "strict_artifact_ids",
            "context_artifact_ids",
        ):
            source_ids = getattr(snapshot, field)
            payload[field] = (
                None
                if source_ids is None
                else [
                    artifacts[artifact_id] for artifact_id in source_ids if artifact_id in artifacts
                ]
            )
        payload["visual_posters"] = {
            artifacts[video_id]: artifacts[poster_id]
            for video_id, poster_id in snapshot.visual_posters.items()
            if video_id in artifacts and poster_id in artifacts
        }
        for field, id_field in (
            ("profile", "profile_id"),
            ("vision_profile", "vision_profile_id"),
            ("verification_profile", None),
        ):
            profile = payload[field]
            old_id = (
                getattr(snapshot, id_field)
                if id_field is not None
                else profile["id"]
                if profile
                else None
            )
            new_profile_id = dependencies.profile_ids.get(old_id or "") if dependencies else None
            if old_id is not None and new_profile_id is None:
                new_profile_id = missing.setdefault(old_id, new_id("missing"))
            if id_field is not None:
                payload[id_field] = new_profile_id
            if profile is not None:
                profile["id"] = new_profile_id
                profile["install"] = None
        old_revision = snapshot.workflow_revision_id
        new_revision = dependencies.revision_ids.get(old_revision or "") if dependencies else None
        if old_revision is not None and new_revision is None:
            new_revision = missing.setdefault(old_revision, new_id("missing"))
        payload["workflow_revision_id"] = new_revision
        workflow = payload["workflow"]
        if workflow is not None:
            workflow["id"] = new_revision
            workflow["workflow_id"] = (
                dependencies.workflow_ids.get(snapshot.workflow.workflow_id)
                if dependencies and snapshot.workflow
                else None
            ) or new_id("missing")
            workflow["trusted"] = False
        for field in ("preset", "preset_layers", "auxiliary_assets"):
            payload[field] = copy.deepcopy(run.provenance_json.get(field))
        payload["preset_layers"] = payload["preset_layers"] or []
        payload["auxiliary_assets"] = payload["auxiliary_assets"] or {}
        imported = AcceptedContext.model_validate(payload)
        payload = imported.model_dump(mode="json")
        digest = _digest(payload)
        session.add(RunContextSnapshot(run_id=run.id, payload_json=payload, sha256=digest))
        session.flush()
        for artifact_id in imported.artifact_ids:
            session.add(RunContextArtifact(run_id=run.id, artifact_id=artifact_id))
        run.provenance_json = {**run.provenance_json, "accepted_context_sha256": digest}
    session.flush()
    # Revision metadata was created before the final imported snapshot digest existed.
    for revision in session.scalars(
        select(ResponseRevision).where(ResponseRevision.run_id.in_(run_ids.values()))
    ):
        revision_run = session.get(Run, revision.run_id)
        if revision_run is not None:
            for part in revision.parts:
                if part.type == "generation_metadata":
                    part.metadata_json = {
                        **part.metadata_json,
                        "run_id": revision_run.id,
                        "provenance": copy.deepcopy(revision_run.provenance_json),
                    }
