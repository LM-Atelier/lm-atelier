"""Relative queue moves compare visible owners and the neighbours observed with them."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, StrictStr, model_validator

from .schemas import ApiModel, QueueControlCommand

QueueLane = Literal["generation", "transfer", "install"]


class QueueOrderOwner(ApiModel):
    type: Literal["work_plan", "job"]
    id: StrictStr = Field(min_length=1, max_length=40)


class QueueOrderNeighbours(ApiModel):
    before: QueueOrderOwner | None
    after: QueueOrderOwner | None


class QueueOrderCommand(QueueControlCommand):
    cohort_id: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    owner: QueueOrderOwner
    before: QueueOrderOwner | None = None
    after: QueueOrderOwner | None = None
    expected_item_neighbors: QueueOrderNeighbours
    expected_anchor_neighbors: QueueOrderNeighbours

    @model_validator(mode="after")
    def one_placement(self) -> Self:
        if (self.before is None) == (self.after is None):
            raise ValueError("Choose exactly one neighbouring item.")
        if self.owner == (self.before or self.after):
            raise ValueError("Choose a different neighbouring item.")
        return self


class QueueOrderItemOut(ApiModel):
    owner: QueueOrderOwner
    label: str
    queued_at: datetime
    priority: int | None
    cohort_id: str | None
    position: int | None = Field(ge=1)
    cohort_length: int = Field(ge=0)
    neighbors: QueueOrderNeighbours
    before_neighbors: QueueOrderNeighbours | None = None
    after_neighbors: QueueOrderNeighbours | None = None
    unavailable_reason: (
        Literal["lane-busy", "held", "blocked", "mixed-resources", "unsupported"] | None
    )


class QueueOrderPageOut(ApiModel):
    lane: QueueLane
    revision: int = Field(ge=0)
    items: list[QueueOrderItemOut]
    total: int = Field(ge=0)
    next_cursor: str | None


class QueueOrderResultOut(ApiModel):
    lane: QueueLane
    revision: int = Field(ge=1)
    owner: QueueOrderOwner
