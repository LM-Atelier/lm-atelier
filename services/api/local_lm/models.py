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
    routing_mode: Mapped[str] = mapped_column(String(16), default=RoutingMode.AUTO.value)
    confirm_uncertain_media: Mapped[bool] = mapped_column(Boolean, default=True)
    active_chat_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    active_image_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    active_video_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    active_head_message_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    generation_settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    generation_preset_ids_json: Mapped[dict[str, str | None]] = mapped_column(JSON, default=dict)

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
    operation: Mapped[str] = mapped_column(String(32), default=Operation.TEXT.value)
    status: Mapped[str] = mapped_column(String(16), default=RunStatus.PENDING.value, index=True)
    standalone_prompt: Mapped[str] = mapped_column(Text, default="")
    profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
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
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    phase: Mapped[str] = mapped_column(String(120), default="queued")
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
