from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .domain import ArtifactKind, JobStatus, utcnow
from .models import (
    Artifact,
    Chat,
    Job,
    ModelCapabilityEvidence,
    ModelInstall,
    ModelProfile,
    Run,
    SetupVerification,
    WorkflowRevision,
)
from .prompt_helpers import prompt_preview_settings
from .schemas import SettingField

if TYPE_CHECKING:
    from .artifacts import ArtifactStore

Role = Literal["chat", "image", "video"]

SETUP_VERIFICATION_SCOPE = "setup_verification"
SETUP_VERIFICATION_VERSION = "setup-generation-v1"
ACTIVE_VERIFICATION_STATES = {"queued", "running"}

_PROMPTS: dict[Role, str] = {
    "chat": "Reply with exactly the single word ready.",
    "image": "A single solid blue circle centered on a plain white background.",
    "video": "A single solid blue circle moving slowly across a plain white background.",
}
_CHAT_OUTPUT_KEYS = {
    "max_tokens",
    "max_new_tokens",
    "maximum_output_tokens",
    "output_tokens",
    "num_predict",
}


def setup_verification_prompt(role: Role) -> str:
    return _PROMPTS[role]


def setup_verification_settings(
    fields: Sequence[SettingField],
    role: Role,
) -> dict[str, int | float]:
    if role != "chat":
        return prompt_preview_settings(list(fields))
    settings: dict[str, int | float] = {}
    for field in fields:
        normalized = field.key.strip().lower().replace("-", "_")
        if (
            not field.available
            or field.scope == "load"
            or field.type not in {"integer", "number"}
            or normalized not in _CHAT_OUTPUT_KEYS
        ):
            continue
        minimum = field.minimum if field.minimum is not None else 1
        maximum = field.maximum if field.maximum is not None else 8
        value = min(max(8, minimum), maximum)
        settings[field.key] = round(value) if field.type == "integer" else value
    return settings


def profile_fingerprint(profile: ModelProfile) -> str:
    payload = {
        "role": profile.role,
        "engine": profile.engine,
        "model_install_id": profile.model_install_id,
        "load_settings": profile.load_settings_json,
        "request_settings": profile.request_settings_json,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def verification_evidence_key(
    role: Role,
    install: ModelInstall,
    profile: ModelProfile,
    workflow: WorkflowRevision | None,
    capability_evidence: ModelCapabilityEvidence,
) -> str:
    payload = {
        "version": SETUP_VERIFICATION_VERSION,
        "role": role,
        "install_id": install.id,
        "profile_id": profile.id,
        "profile_fingerprint": profile_fingerprint(profile),
        "workflow_revision_id": workflow.id if workflow else None,
        "activation_evidence_key": capability_evidence.evidence_key,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def current_setup_verification(
    session: Session,
    role: Role,
    install: ModelInstall,
    profile: ModelProfile,
    workflow: WorkflowRevision | None,
    capability_evidence: ModelCapabilityEvidence,
) -> SetupVerification | None:
    key = verification_evidence_key(
        role,
        install,
        profile,
        workflow,
        capability_evidence,
    )
    return session.scalar(select(SetupVerification).where(SetupVerification.evidence_key == key))


def setup_verification_for_chat(
    session: Session,
    chat_id: str,
) -> SetupVerification | None:
    return session.scalar(select(SetupVerification).where(SetupVerification.chat_id == chat_id))


def mark_setup_verification_running(
    session: Session,
    chat_id: str,
    *,
    run_id: str,
    job_id: str,
) -> None:
    verification = setup_verification_for_chat(session, chat_id)
    if not verification or verification.state not in ACTIVE_VERIFICATION_STATES:
        return
    verification.state = "running"
    verification.run_id = run_id
    verification.job_id = job_id
    verification.started_at = verification.started_at or utcnow()


def finalize_setup_verification(
    session: Session,
    artifacts: ArtifactStore,
    chat_id: str,
    job: Job,
) -> bool:
    verification = setup_verification_for_chat(session, chat_id)
    if not verification:
        return False

    result = job.result_json if isinstance(job.result_json, dict) else {}
    has_output = (
        int(result.get("characters", 0) or 0) > 0
        if verification.role == "chat"
        else bool(result.get("artifact_ids"))
    )
    if job.status == JobStatus.COMPLETE.value and has_output:
        verification.state = "ready"
        verification.failure_code = None
    else:
        verification.state = "failed"
        verification.failure_code = (
            "empty_generation"
            if job.status == JobStatus.COMPLETE.value
            else {
                JobStatus.CANCELLED.value: "generation_cancelled",
                JobStatus.INTERRUPTED.value: "application_restarted",
            }.get(job.status, "generation_failed")
        )
    verification.completed_at = utcnow()

    input_artifact_id = verification.input_artifact_id
    artifacts.delete_chat_generated_media(session, chat_id)
    if input_artifact_id and (artifact := session.get(Artifact, input_artifact_id)):
        artifacts.delete_library_artifact(session, artifact, release_membership=True)

    if chat := session.get(Chat, chat_id):
        session.delete(chat)
    session.delete(job)
    verification.chat_id = None
    verification.run_id = None
    verification.job_id = None
    verification.input_artifact_id = None
    return True


def recover_terminal_setup_verifications(
    session: Session,
    artifacts: ArtifactStore,
) -> None:
    verifications = session.scalars(
        select(SetupVerification).where(SetupVerification.state.in_(ACTIVE_VERIFICATION_STATES))
    ).all()
    for verification in verifications:
        if not verification.chat_id:
            verification.state = "failed"
            verification.failure_code = "application_restarted"
            verification.completed_at = utcnow()
            continue
        job = (
            session.get(Job, verification.job_id)
            if verification.job_id
            else session.scalar(
                select(Job)
                .join(Run, Job.run_id == Run.id)
                .where(Run.chat_id == verification.chat_id)
                .order_by(Job.created_at.desc())
                .limit(1)
            )
        )
        if job and job.status in {
            JobStatus.COMPLETE.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.INTERRUPTED.value,
        }:
            finalize_setup_verification(
                session,
                artifacts,
                verification.chat_id,
                job,
            )
        elif not job:
            verification.state = "failed"
            verification.failure_code = "application_restarted"
            verification.completed_at = utcnow()
            if verification.input_artifact_id and (
                artifact := session.get(Artifact, verification.input_artifact_id)
            ):
                artifacts.delete_library_artifact(session, artifact, release_membership=True)
            if chat := session.get(Chat, verification.chat_id):
                session.delete(chat)
            verification.chat_id = None
            verification.input_artifact_id = None


def synthetic_setup_image(verification_id: str) -> bytes:
    width = 16
    height = 16
    scanline = b"\x00" + (b"\x1f\x66\xcc" * width)
    pixels = scanline * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(
                b"tEXt",
                f"lm-atelier-verification\x00{verification_id}".encode("ascii"),
            ),
            chunk(b"IDAT", zlib.compress(pixels)),
            chunk(b"IEND", b""),
        )
    )


def ingest_synthetic_setup_image(
    session: Session,
    artifacts: ArtifactStore,
    verification_id: str,
) -> Artifact:
    return artifacts.ingest_bytes(
        session,
        synthetic_setup_image(verification_id),
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name="setup-verification.png",
        metadata={"setup_verification_id": verification_id},
    )
