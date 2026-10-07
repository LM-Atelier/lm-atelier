"""Bind one bounded selection to an all-or-nothing recovery command."""

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from .recovery_v1 import (
    RecoveryCommandV1,
    RecoveryContract,
    RecoveryDigest,
    RecoveryIdentity,
    RecoveryImpactV1,
    RecoveryResultV1,
)

BatchAction = Literal["restore", "purge"]


class RecoveryBatchSelectionV1(RecoveryContract):
    deletion_ids: Annotated[tuple[RecoveryIdentity, ...], Field(min_length=1, max_length=20)]
    action: BatchAction
    restore_unfiled: StrictBool = False

    @model_validator(mode="after")
    def unique_selection(self) -> Self:
        if len(set(self.deletion_ids)) != len(self.deletion_ids):
            raise ValueError("Choose each deleted item once.")
        if self.action == "purge" and self.restore_unfiled:
            raise ValueError("Unfiled restoration applies only to Restore.")
        return self


class RecoveryBatchMemberV1(RecoveryContract):
    deletion_id: RecoveryIdentity
    display_label: Annotated[str, Field(min_length=1, max_length=240)]
    purge_after: datetime
    impact: RecoveryImpactV1


class RecoveryBatchPreviewV1(RecoveryContract):
    batch_id: RecoveryIdentity
    revision: RecoveryDigest
    impact_sha256: RecoveryDigest
    action: BatchAction
    policy: Literal["all-or-nothing"] = "all-or-nothing"
    restore_unfiled: StrictBool
    available: StrictBool
    expires_at: datetime
    items: Annotated[tuple[RecoveryBatchMemberV1, ...], Field(min_length=1, max_length=20)]


class RecoveryBatchApplyV1(RecoveryCommandV1):
    acknowledgement: Literal["permanently-delete"] | None = None


class RecoveryBatchResultV1(RecoveryContract):
    batch_id: RecoveryIdentity
    action: BatchAction
    policy: Literal["all-or-nothing"] = "all-or-nothing"
    reclaimed_bytes: StrictInt = Field(ge=0)
    results: Annotated[tuple[RecoveryResultV1, ...], Field(min_length=1, max_length=20)]
