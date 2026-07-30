from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utcnow() -> datetime:
    return datetime.now(UTC)


def elapsed_milliseconds(started_at: datetime, completed_at: datetime) -> int:
    """Return elapsed wall time while treating SQLite-naive timestamps as UTC."""
    normalized_start = (
        started_at.replace(tzinfo=UTC) if started_at.tzinfo is None else started_at.astimezone(UTC)
    )
    normalized_end = (
        completed_at.replace(tzinfo=UTC)
        if completed_at.tzinfo is None
        else completed_at.astimezone(UTC)
    )
    return max(0, int((normalized_end - normalized_start).total_seconds() * 1000))


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageStatus(StrEnum):
    COMPLETE = "complete"
    PENDING = "pending"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PartType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    ATTACHMENT = "attachment"
    PROGRESS = "progress"
    ERROR = "error"
    GENERATION_METADATA = "generation_metadata"


class RoutingMode(StrEnum):
    AUTO = "auto"
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class Operation(StrEnum):
    TEXT = "text"
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"


class RunStatus(StrEnum):
    PENDING = "pending"
    ROUTING = "routing"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobKind(StrEnum):
    CHAT = "chat"
    IMAGE = "image"
    VIDEO = "video"
    EDIT_VERIFY = "edit_verify"
    ACTIVATE = "activate"
    DOWNLOAD = "download"
    EXPORT = "export"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class ArtifactKind(StrEnum):
    MODEL = "model"
    IMAGE = "image"
    VIDEO = "video"
    THUMBNAIL = "thumbnail"
    INPUT = "input"
    EXPORT = "export"
    OTHER = "other"


class ModelRole(StrEnum):
    CHAT = "chat"
    IMAGE = "image"
    VIDEO = "video"


class CompatibilityLevel(StrEnum):
    TESTED = "tested"
    LIKELY = "likely"
    ADVANCED = "advanced_import"
    UNSUPPORTED = "unsupported"
