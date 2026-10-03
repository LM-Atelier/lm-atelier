"""Describe recoverable resources without copying their content into a recovery record."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

RecoveryIdentity = Annotated[StrictStr, Field(min_length=1, max_length=128)]
RecoveryDigest = Annotated[
    StrictStr, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
]
RecoveryOperationKey = Annotated[
    StrictStr, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
]
RecoveryCursor = Annotated[StrictStr, Field(min_length=1, max_length=2048)]


class RecoveryKind(StrEnum):
    CHAT = "chat"
    PROJECT = "project"
    MEDIA_LIBRARY_ENTRY = "media_library_entry"
    WORKFLOW_FAMILY = "workflow_family"


class RecoveryState(StrEnum):
    RECOVERABLE = "recoverable"
    RESTORING = "restoring"
    PURGING = "purging"
    BLOCKED = "blocked"
    PURGED = "purged"


class RecoveryAction(StrEnum):
    TRASH = "trash"
    RESTORE = "restore"
    PURGE = "purge"


class RecoveryConflict(StrEnum):
    ORIGINAL_PROJECT_MISSING = "original_project_missing"
    SUBJECT_MISSING = "subject_missing"
    ACTIVE_WORK = "active_work"
    ACTIVE_SELECTION = "active_selection"
    RECORD_CHANGED = "record_changed"


class RecoveryContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)


class RecoveryCountsV1(RecoveryContract):
    chats: StrictInt = Field(default=0, ge=0)
    messages: StrictInt = Field(default=0, ge=0)
    message_parts: StrictInt = Field(default=0, ge=0)
    response_revisions: StrictInt = Field(default=0, ge=0)
    runs: StrictInt = Field(default=0, ge=0)
    jobs: StrictInt = Field(default=0, ge=0)
    work_plans: StrictInt = Field(default=0, ge=0)
    workflow_families: StrictInt = Field(default=0, ge=0)
    workflow_definitions: StrictInt = Field(default=0, ge=0)
    workflow_revisions: StrictInt = Field(default=0, ge=0)
    references: StrictInt = Field(default=0, ge=0)
    artifacts: StrictInt = Field(default=0, ge=0)
    active_work: StrictInt = Field(default=0, ge=0)
    retained_bytes: StrictInt = Field(default=0, ge=0)
    reclaimable_bytes: StrictInt = Field(default=0, ge=0)


class RecoveryLocationV1(RecoveryContract):
    project_id: RecoveryIdentity | None = None
    project_label: Annotated[StrictStr, Field(max_length=200)] | None = None


class RecoveryImpactV1(RecoveryContract):
    kind: RecoveryKind
    subject_id: RecoveryIdentity
    revision: RecoveryDigest
    impact_sha256: RecoveryDigest
    counts: RecoveryCountsV1
    conflicts: tuple[RecoveryConflict, ...] = ()
    available_actions: tuple[RecoveryAction, ...] = ()
    delete_generated_media: StrictBool = False
    reclaimed_bytes: StrictInt = Field(default=0, ge=0)


class RecoveryItemV1(RecoveryContract):
    deletion_id: RecoveryIdentity
    kind: RecoveryKind
    subject_id: RecoveryIdentity
    display_label: Annotated[StrictStr, Field(min_length=1, max_length=240)]
    original_location: RecoveryLocationV1
    deleted_at: datetime
    purge_after: datetime
    state: RecoveryState
    revision: RecoveryDigest
    counts: RecoveryCountsV1
    restore_conflicts: tuple[RecoveryConflict, ...] = ()
    delete_generated_media: StrictBool = False

    @field_validator("deleted_at", "purge_after")
    @classmethod
    def utc_dates(cls, value: datetime) -> datetime:
        """SQLite stores UTC without an offset; responses retain that meaning explicitly."""
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @model_validator(mode="after")
    def recovery_window(self) -> Self:
        if self.purge_after <= self.deleted_at:
            raise ValueError("Recovery expiry must be after deletion.")
        return self


class RecoveryPageV1(RecoveryContract):
    items: tuple[RecoveryItemV1, ...]
    next_cursor: RecoveryCursor | None = None


class RecoveryCommandV1(RecoveryContract):
    expected_revision: RecoveryDigest
    impact_sha256: RecoveryDigest
    operation_key: RecoveryOperationKey


class TrashChatV1(RecoveryCommandV1):
    delete_generated_media: StrictBool = False


class RestoreRecoveryV1(RecoveryCommandV1):
    restore_unfiled: StrictBool = False


class PurgeRecoveryV1(RecoveryCommandV1):
    acknowledgement: Literal["permanently-delete"]


class DeletionQuarantineV1(RecoveryContract):
    kind: RecoveryKind
    subject_id: RecoveryIdentity
    action: RecoveryAction
    expected_revision: RecoveryDigest
    impact_sha256: RecoveryDigest
    operation_key: RecoveryOperationKey
    delete_generated_media: StrictBool = False


class RecoveryResultV1(RecoveryContract):
    deletion_id: RecoveryIdentity
    kind: RecoveryKind
    subject_id: RecoveryIdentity
    action: RecoveryAction
    replayed: StrictBool = False
    reclaimed_bytes: StrictInt = Field(default=0, ge=0)
