from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base
from .domain import (
    ArtifactKind,
    CompatibilityLevel,
    JobKind,
    JobStatus,
    MessageRole,
    MessageStatus,
    ModelRole,
    Operation,
    PartType,
    RoutingMode,
    RunStatus,
    new_id,
    utcnow,
)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("proj"))
    name: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    instructions: Mapped[str] = mapped_column(Text, default="")
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    image_workflow_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_revisions.id", ondelete="SET NULL"), nullable=True
    )
    video_workflow_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_revisions.id", ondelete="SET NULL"), nullable=True
    )
    generation_settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    generation_preset_ids_json: Mapped[dict[str, str | None]] = mapped_column(JSON, default=dict)

    chats: Mapped[list[Chat]] = relationship(back_populates="project")


class Chat(TimestampMixin, Base):
    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("chat"))
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(240), default="New chat")
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    scope: Mapped[str] = mapped_column(String(24), default="standard", index=True)
    draft_prompt: Mapped[str] = mapped_column(Text, default="")
    routing_mode: Mapped[str] = mapped_column(String(16), default=RoutingMode.AUTO.value)
    confirm_uncertain_media: Mapped[bool] = mapped_column(Boolean, default=True)
    active_chat_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    active_vision_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    active_image_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    active_video_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    active_head_message_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    generation_settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    generation_preset_ids_json: Mapped[dict[str, str | None]] = mapped_column(JSON, default=dict)
    vision_settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Where a forked thread came from, empty for chats created directly.
    origin_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, server_default="{}")

    project: Mapped[Project | None] = relationship(back_populates="chats")
    messages: Mapped[list[Message]] = relationship(
        back_populates="chat", cascade="all, delete-orphan", order_by="Message.created_at"
    )
    runs: Mapped[list[Run]] = relationship(back_populates="chat", cascade="all, delete-orphan")


class Message(TimestampMixin, Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("msg"))
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"), nullable=True, index=True
    )
    role: Mapped[str] = mapped_column(String(16), default=MessageRole.USER.value)
    status: Mapped[str] = mapped_column(String(16), default=MessageStatus.COMPLETE.value)
    transcript_visible: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default="1",
        index=True,
    )
    active_response_revision_id: Mapped[str | None] = mapped_column(String(40), nullable=True)

    chat: Mapped[Chat] = relationship(back_populates="messages")
    parts: Mapped[list[MessagePart]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="MessagePart.position",
    )
    response_revisions: Mapped[list[ResponseRevision]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="ResponseRevision.sequence",
        foreign_keys="ResponseRevision.message_id",
    )


class MessagePart(TimestampMixin, Base):
    __tablename__ = "message_parts"
    __table_args__ = (UniqueConstraint("message_id", "position", name="uq_message_part_position"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("part"))
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(32), default=PartType.TEXT.value)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    message: Mapped[Message] = relationship(back_populates="parts")
    artifact: Mapped[Artifact | None] = relationship()


class WorkPlan(TimestampMixin, Base):
    __tablename__ = "work_plans"
    __table_args__ = (
        UniqueConstraint(
            "chat_id",
            "transcript_sequence",
            name="uq_work_plan_transcript_sequence",
        ),
        UniqueConstraint(
            "chat_id",
            "idempotency_key",
            name="uq_work_plan_chat_id_idempotency_key",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("plan"))
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    source_action: Mapped[str] = mapped_column(String(32), default="send")
    persistence_scope: Mapped[str] = mapped_column(String(16), default="durable")
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value, index=True)
    context_head_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    transcript_sequence: Mapped[int] = mapped_column(Integer)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    planner_version: Mapped[str] = mapped_column(String(32), default="legacy-turn-v1")
    failure_policy: Mapped[str] = mapped_column(String(32), default="stop_dependents")
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    steps: Mapped[list[WorkStep]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="WorkStep.ordinal",
    )


class WorkStep(TimestampMixin, Base):
    __tablename__ = "work_steps"
    __table_args__ = (
        UniqueConstraint("plan_id", "ordinal", name="uq_work_step_ordinal"),
        UniqueConstraint("run_id", name="uq_work_step_run"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("step"))
    plan_id: Mapped[str] = mapped_column(
        ForeignKey("work_plans.id", ondelete="CASCADE"),
        index=True,
    )
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    display_group: Mapped[str | None] = mapped_column(String(80), nullable=True)
    operation: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value, index=True)
    prompt: Mapped[str] = mapped_column(Text, default="")
    profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    workflow_revision_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    input_bindings_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    output_contract_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    queue_class: Mapped[str] = mapped_column(String(32), default="interactive_compute")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    plan: Mapped[WorkPlan] = relationship(back_populates="steps")


class WorkStepDependency(Base):
    __tablename__ = "work_step_dependencies"

    step_id: Mapped[str] = mapped_column(
        ForeignKey("work_steps.id", ondelete="CASCADE"),
        primary_key=True,
    )
    depends_on_step_id: Mapped[str] = mapped_column(
        ForeignKey("work_steps.id", ondelete="CASCADE"),
        primary_key=True,
    )


class Run(TimestampMixin, Base):
    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint(
            "chat_id",
            "idempotency_key",
            name="uq_runs_chat_id_idempotency_key",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("run"))
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    user_message_id: Mapped[str] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    assistant_message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), unique=True
    )
    work_plan_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_plans.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    work_step_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_steps.id", ondelete="SET NULL"),
        nullable=True,
        unique=True,
    )
    operation: Mapped[str] = mapped_column(String(32), default=Operation.TEXT.value)
    status: Mapped[str] = mapped_column(String(16), default=RunStatus.PENDING.value, index=True)
    standalone_prompt: Mapped[str] = mapped_column(Text, default="")
    profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    vision_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    workflow_revision_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    chat: Mapped[Chat] = relationship(back_populates="runs")


class ResponseRevision(TimestampMixin, Base):
    __tablename__ = "response_revisions"
    __table_args__ = (
        UniqueConstraint("message_id", "sequence", name="uq_response_revision_sequence"),
        UniqueConstraint("run_id", name="uq_response_revision_run"),
        Index(
            "uq_response_revision_pending_message",
            "message_id",
            unique=True,
            sqlite_where=text("status = 'pending'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("rev"))
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default=MessageStatus.PENDING.value, index=True)

    message: Mapped[Message] = relationship(
        back_populates="response_revisions",
        foreign_keys=[message_id],
    )
    parts: Mapped[list[ResponseRevisionPart]] = relationship(
        back_populates="revision",
        cascade="all, delete-orphan",
        order_by="ResponseRevisionPart.position",
    )


class ResponseRevisionPart(TimestampMixin, Base):
    __tablename__ = "response_revision_parts"
    __table_args__ = (
        UniqueConstraint(
            "response_revision_id",
            "position",
            name="uq_response_revision_part_position",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("revpart"))
    response_revision_id: Mapped[str] = mapped_column(
        ForeignKey("response_revisions.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(32), default=PartType.TEXT.value)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    revision: Mapped[ResponseRevision] = relationship(back_populates="parts")
    artifact: Mapped[Artifact | None] = relationship()


class TurnCreationClaim(Base):
    """Short-lived database lease that deduplicates turn planning across orchestrators."""

    __tablename__ = "turn_creation_claims"
    __table_args__ = (
        UniqueConstraint(
            "chat_id",
            "idempotency_key",
            name="uq_turn_creation_claim_chat_id_idempotency_key",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("claim"))
    chat_id: Mapped[str] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    owner_token: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Artifact(TimestampMixin, Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(24), default=ArtifactKind.OTHER.value)
    media_type: Mapped[str] = mapped_column(String(120))
    size_bytes: Mapped[int] = mapped_column(Integer)
    relative_path: Mapped[str] = mapped_column(Text, unique=True)
    original_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ModelSource(TimestampMixin, Base):
    __tablename__ = "model_sources"
    __table_args__ = (
        UniqueConstraint("provider", "remote_id", "revision", name="uq_model_source"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("source"))
    provider: Mapped[str] = mapped_column(String(32), default="huggingface")
    remote_id: Mapped[str] = mapped_column(String(500))
    revision: Mapped[str] = mapped_column(String(200), default="main")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class InstallPlan(TimestampMixin, Base):
    __tablename__ = "install_plans"
    __table_args__ = (
        UniqueConstraint("plan_hash", name="uq_install_plan_hash"),
        Index("ix_install_plan_source", "provider", "remote_id", "revision", "role"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("plan"))
    provider: Mapped[str] = mapped_column(String(32), default="huggingface")
    remote_id: Mapped[str] = mapped_column(String(500))
    revision: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(16))
    engine: Mapped[str] = mapped_column(String(32))
    architecture: Mapped[str | None] = mapped_column(String(200), nullable=True)
    family: Mapped[str | None] = mapped_column(String(100), nullable=True)
    plan_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    resolver_version: Mapped[str] = mapped_column(String(40))
    compatibility: Mapped[str] = mapped_column(String(40))
    artifacts_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    runtime_contract_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    activation_probe_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(24), default="planned", index=True)
    failure_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class ModelInstall(TimestampMixin, Base):
    __tablename__ = "model_installs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("model"))
    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_sources.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(300), index=True)
    role: Mapped[str] = mapped_column(String(16), default=ModelRole.CHAT.value)
    engine: Mapped[str] = mapped_column(String(32))
    local_path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    compatibility: Mapped[str] = mapped_column(
        String(24), default=CompatibilityLevel.ADVANCED.value
    )
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class ModelAssetInstall(TimestampMixin, Base):
    __tablename__ = "model_asset_installs"

    id: Mapped[str] = mapped_column(
        String(40),
        primary_key=True,
        default=lambda: new_id("asset"),
    )
    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_sources.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(300), index=True)
    kind: Mapped[str] = mapped_column(String(40), index=True)
    family: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    local_path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    use_case: Mapped[str] = mapped_column(Text, default="")
    auto_apply: Mapped[bool] = mapped_column(Boolean, default=False)
    default_model_strength: Mapped[float] = mapped_column(Float, default=1.0)
    default_clip_strength: Mapped[float] = mapped_column(Float, default=1.0)
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ModelComponentManifest(TimestampMixin, Base):
    __tablename__ = "model_component_manifests"
    __table_args__ = (
        UniqueConstraint(
            "model_install_id",
            "relative_path",
            name="uq_model_component_path",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(40),
        primary_key=True,
        default=lambda: new_id("component"),
    )
    model_install_id: Mapped[str] = mapped_column(
        ForeignKey("model_installs.id", ondelete="CASCADE"),
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(40))
    relative_path: Mapped[str] = mapped_column(Text)
    target_folder: Mapped[str] = mapped_column(String(80))
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    required: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ModelCapabilityEvidence(TimestampMixin, Base):
    __tablename__ = "model_capability_evidence"
    __table_args__ = (
        UniqueConstraint(
            "model_install_id",
            "evidence_key",
            name="uq_model_capability_evidence_install_key",
        ),
        Index("ix_model_capability_evidence_install_result", "model_install_id", "result"),
    )

    id: Mapped[str] = mapped_column(
        String(40),
        primary_key=True,
        default=lambda: new_id("evidence"),
    )
    model_install_id: Mapped[str] = mapped_column(
        ForeignKey("model_installs.id", ondelete="CASCADE"),
        index=True,
    )
    evidence_key: Mapped[str] = mapped_column(String(64), index=True)
    result: Mapped[str] = mapped_column(String(24))
    component_hashes_json: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    runtime_build: Mapped[str] = mapped_column(String(200))
    adapter_contract_version: Mapped[int] = mapped_column(Integer)
    launch_contract_version: Mapped[str] = mapped_column(String(40))
    workflow_contract_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    hardware_class: Mapped[str] = mapped_column(String(200))
    # What the proving machine offered, so a proof survives a driver update or a
    # PATH change. Null on rows written before envelopes existed, which fall back
    # to comparing `hardware_class` for equality.
    hardware_envelope_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    probe_version: Mapped[str] = mapped_column(String(40))
    failure_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    probed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SetupVerification(TimestampMixin, Base):
    __tablename__ = "setup_verifications"
    __table_args__ = (
        UniqueConstraint("evidence_key", name="uq_setup_verification_evidence_key"),
        Index("ix_setup_verifications_role_state", "role", "state"),
    )

    id: Mapped[str] = mapped_column(
        String(40),
        primary_key=True,
        default=lambda: new_id("verify"),
    )
    role: Mapped[str] = mapped_column(String(16))
    evidence_key: Mapped[str] = mapped_column(String(64), unique=True)
    state: Mapped[str] = mapped_column(String(24), default="queued")
    model_install_id: Mapped[str] = mapped_column(
        ForeignKey("model_installs.id", ondelete="CASCADE"),
    )
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("model_profiles.id", ondelete="CASCADE"),
    )
    workflow_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_revisions.id", ondelete="CASCADE"),
        nullable=True,
    )
    chat_id: Mapped[str | None] = mapped_column(String(40), nullable=True, unique=True, index=True)
    run_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    input_artifact_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelProfile(TimestampMixin, Base):
    __tablename__ = "model_profiles"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("profile"))
    model_install_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_installs.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(200), index=True)
    use_case: Mapped[str] = mapped_column(Text, default="")
    role: Mapped[str] = mapped_column(String(16), default=ModelRole.CHAT.value)
    engine: Mapped[str] = mapped_column(String(32))
    load_settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    request_settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)


class GenerationPreset(TimestampMixin, Base):
    __tablename__ = "generation_presets"
    __table_args__ = (UniqueConstraint("role", "name", name="uq_preset_role_name"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("preset"))
    name: Mapped[str] = mapped_column(String(200), index=True)
    role: Mapped[str] = mapped_column(String(16), default=ModelRole.CHAT.value, index=True)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)


class WorkflowDefinition(TimestampMixin, Base):
    __tablename__ = "workflow_definitions"

    id: Mapped[str] = mapped_column(
        String(40), primary_key=True, default=lambda: new_id("workflow")
    )
    name: Mapped[str] = mapped_column(String(240), index=True)
    operation: Mapped[str] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(Text, default="")
    current_revision_id: Mapped[str | None] = mapped_column(String(40), nullable=True)

    revisions: Mapped[list[WorkflowRevision]] = relationship(
        back_populates="definition",
        cascade="all, delete-orphan",
        foreign_keys="WorkflowRevision.workflow_id",
    )


class WorkflowRevision(TimestampMixin, Base):
    __tablename__ = "workflow_revisions"
    __table_args__ = (UniqueConstraint("workflow_id", "version", name="uq_workflow_version"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("wfrev"))
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflow_definitions.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    engine: Mapped[str] = mapped_column(String(32), default="comfyui")
    engine_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    ui_graph_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    api_graph_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    input_schema_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    dependencies_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Identifies what this revision executes, so capability evidence survives a
    # compiler change that does not alter the compiled output.
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    trusted: Mapped[bool] = mapped_column(Boolean, default=False)

    definition: Mapped[WorkflowDefinition] = relationship(
        back_populates="revisions", foreign_keys=[workflow_id]
    )


class CustomNodeInstall(TimestampMixin, Base):
    __tablename__ = "custom_node_installs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("node"))
    name: Mapped[str] = mapped_column(String(240), index=True)
    source_url: Mapped[str] = mapped_column(String(1000), unique=True)
    revision: Mapped[str] = mapped_column(String(40))
    previous_revision: Mapped[str | None] = mapped_column(String(40), nullable=True)
    installed_path: Mapped[str] = mapped_column(Text)
    tree_hash: Mapped[str] = mapped_column(String(64))
    trusted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    security_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Job(TimestampMixin, Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("job"))
    kind: Mapped[str] = mapped_column(String(16), default=JobKind.CHAT.value)
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value, index=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    work_plan_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_plans.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    work_step_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_steps.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    phase: Mapped[str] = mapped_column(String(120), default="queued")
    progress_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    queue_resource: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    queue_group: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    queue_priority: Mapped[int] = mapped_column(Integer, default=0, index=True)
    queue_ticket: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    enqueued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    claim_owner: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    claim_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    cancellable: Mapped[bool] = mapped_column(Boolean, default=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AppSetting(TimestampMixin, Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    value_json: Mapped[Any] = mapped_column(JSON)
