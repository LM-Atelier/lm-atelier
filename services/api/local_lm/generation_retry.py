"""Persist bounded media retry choices and account for each reserved attempt."""

from typing import Any

from sqlalchemy import update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.dml import Insert, Update

from .domain import Operation, utcnow
from .models import AppSetting
from .schemas import GenerationRetryPolicyOut, GenerationRetryPolicyUpdate

POLICY_KEY = "generation_failure_retries"
MEDIA_OPERATIONS = frozenset(
    operation.value
    for operation in (
        Operation.TEXT_TO_IMAGE,
        Operation.IMAGE_TO_IMAGE,
        Operation.TEXT_TO_VIDEO,
        Operation.IMAGE_TO_VIDEO,
    )
)


class GenerationRetryPolicyConflict(ValueError):
    """The saved retry choice changed after it was read."""


def read_retry_policy(session: Session) -> GenerationRetryPolicyOut:
    row = session.get(AppSetting, POLICY_KEY, populate_existing=True)
    if row is None:
        return GenerationRetryPolicyOut()
    if not isinstance(row.value_json, dict) or set(row.value_json) != {"max_retries", "revision"}:
        raise ValueError("The saved generation retry setting is invalid.")
    policy = GenerationRetryPolicyOut.model_validate(row.value_json)
    if policy.revision < 1:
        raise ValueError("The saved generation retry setting is invalid.")
    return policy


def write_retry_policy(
    session: Session, choice: GenerationRetryPolicyUpdate
) -> GenerationRetryPolicyOut:
    """Save one choice with a revision check; the calling route commits."""
    policy = GenerationRetryPolicyOut(
        max_retries=choice.max_retries, revision=choice.expected_revision + 1
    )
    now = utcnow()
    statement: Insert | Update
    if choice.expected_revision == 0:
        statement = (
            sqlite_insert(AppSetting)
            .values(key=POLICY_KEY, value_json=policy.model_dump(), created_at=now, updated_at=now)
            .on_conflict_do_nothing(index_elements=["key"])
        )
    else:
        statement = (
            update(AppSetting)
            .where(
                AppSetting.key == POLICY_KEY,
                AppSetting.value_json["revision"].as_integer() == choice.expected_revision,
            )
            .values(value_json=policy.model_dump(), updated_at=now)
            .execution_options(synchronize_session=False)
        )
    if getattr(session.execute(statement), "rowcount", 0) != 1:
        raise GenerationRetryPolicyConflict("The generation retry setting changed. Refresh it.")
    return policy


def capture_retry_budget(session: Session, operation: str) -> dict[str, Any]:
    """Freeze a new run's allowance so settings changes cannot extend its retries."""
    return {
        "limit": read_retry_policy(session).max_retries if operation in MEDIA_OPERATIONS else 0,
        "used": 0,
        "pending": False,
    }


def reserve_retry(provenance: dict[str, Any], operation: str) -> dict[str, Any] | None:
    """Spend one additional attempt after the caller has proved failure ownership."""
    value = provenance.get("failure_retries")
    if operation not in MEDIA_OPERATIONS or not isinstance(value, dict):
        return None
    limit = value.get("limit")
    used = value.get("used")
    if (
        type(limit) is not int
        or type(used) is not int
        or not 0 <= used < limit <= 10
        or value.get("pending") is not False
    ):
        return None
    return {"limit": limit, "used": used + 1, "pending": True}


def retry_is_pending(provenance: dict[str, Any]) -> bool:
    value = provenance.get("failure_retries")
    return (
        isinstance(value, dict)
        and type(value.get("limit")) is int
        and type(value.get("used")) is int
        and 0 < value["used"] <= value["limit"] <= 10
        and value.get("pending") is True
    )
