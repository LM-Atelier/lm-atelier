"""Lock and unlock the workspace, and choose whether it locks and which PIN opens it.

Status and unlock are the only routes here a locked workspace still answers: a
locked page reads the status to learn that another tab opened the workspace, and
unlocks it itself. The rest sit behind the lock with everything else.

A wrong PIN is a 403, never a 401: the browser takes a 401 for a lost session
and repeats the request once, which would count every wrong PIN twice. No
answer says how many attempts remain.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from .api_errors import ApiError, api_error
from .db import SessionLocal, get_session
from .schemas import (
    WorkspaceLockPolicyOut,
    WorkspaceLockPolicyWrite,
    WorkspaceLockStatusOut,
    WorkspaceUnlockIn,
)
from .workspace_lock import (
    UnlockAttempts,
    WorkspaceLock,
    WorkspaceLockChanged,
    WorkspaceLockDisabled,
    WorkspaceLockState,
)
from .workspace_lock_policy import (
    PinCheckUnavailable,
    PinVerifier,
    SavedWorkspaceLock,
    WorkspaceLockPolicyStale,
    WorkspaceLockSettingInvalid,
    make_pin_verifier,
    pin_matches,
    pin_within_bounds,
    read_policy,
    write_policy,
)

if TYPE_CHECKING:
    from .main import Services

SessionDep = Annotated[Session, Depends(get_session)]
router = APIRouter(prefix="/privacy")


def _lock(request: Request) -> WorkspaceLock:
    return cast("Services", request.app.state.services).workspace_lock


def _status_out(state: WorkspaceLockState) -> WorkspaceLockStatusOut:
    return WorkspaceLockStatusOut(
        locked=state.locked,
        enabled=state.enabled,
        require_pin=state.require_pin,
        lock_epoch=state.lock_epoch,
    )


def _policy_out(policy: SavedWorkspaceLock) -> WorkspaceLockPolicyOut:
    return WorkspaceLockPolicyOut(
        enabled=policy.enabled, require_pin=policy.pin is not None, revision=policy.revision
    )


def _setting_invalid() -> ApiError:
    return api_error(
        409,
        "workspace-lock-setting-invalid",
        "The saved workspace lock setting cannot be read.",
    )


def _setting_stale(current_revision: int) -> ApiError:
    return api_error(
        409,
        "workspace-lock-setting-stale",
        "The workspace lock setting changed since it was read. Check it and try again.",
        current_revision=current_revision,
    )


def _throttled(seconds: int) -> ApiError:
    error = api_error(
        429,
        "workspace-pin-throttled",
        "Wait a moment before trying the PIN again.",
        retry_after_seconds=seconds,
    )
    error.headers = {"Retry-After": str(seconds)}
    return error


def _refused(wait: int) -> ApiError:
    error = api_error(
        403,
        "workspace-pin-refused",
        "The workspace could not be unlocked.",
        retry_after_seconds=wait,
    )
    if wait:
        error.headers = {"Retry-After": str(wait)}
    return error


def _unavailable() -> ApiError:
    return api_error(
        503,
        "workspace-pin-unavailable",
        "The PIN could not be checked right now. Try again shortly.",
    )


@dataclass(frozen=True)
class _PinCheck:
    unlocked: bool
    retry_after_seconds: int = 0


def _check_saved_pin(attempts: UnlockAttempts, pin: str | None) -> _PinCheck:
    """Check a PIN against the saved setting, in a worker thread with its own session.

    The attempt is settled here rather than by the request, so a request that
    is abandoned mid-check still counts, and the one derivation slot stays
    taken until the derivation it guards has ended.
    """

    try:
        with SessionLocal() as session:
            verifier = read_policy(session).pin
        if verifier is None:
            return _PinCheck(unlocked=True)
        if pin_matches(verifier, pin):
            attempts.succeed()
            return _PinCheck(unlocked=True)
        return _PinCheck(unlocked=False, retry_after_seconds=attempts.fail())
    finally:
        attempts.release()


@router.get("/status", response_model=WorkspaceLockStatusOut)
async def workspace_lock_status(request: Request) -> WorkspaceLockStatusOut:
    """Whether the workspace is locked, from memory alone."""

    return _status_out(_lock(request).status())


@router.post("/unlock", response_model=WorkspaceLockStatusOut)
async def unlock_workspace(
    request: Request, payload: WorkspaceUnlockIn | None = None
) -> WorkspaceLockStatusOut:
    lock = _lock(request)
    state = lock.status()
    if not state.locked:
        return _status_out(state)
    if not lock.setting_readable:
        raise _setting_invalid()
    wait = lock.attempts.admit()
    if wait is not None:
        raise _throttled(wait)
    pin = payload.pin if payload is not None else None
    try:
        check = await asyncio.to_thread(_check_saved_pin, lock.attempts, pin)
    except WorkspaceLockSettingInvalid:
        raise _setting_invalid() from None
    except PinCheckUnavailable:
        raise _unavailable() from None
    if not check.unlocked:
        raise _refused(check.retry_after_seconds)
    try:
        # Opens only the lock this check was admitted for, never a newer one.
        return _status_out(lock.unlock(admitted_epoch=state.lock_epoch))
    except WorkspaceLockChanged:
        raise api_error(
            423,
            "workspace-lock-changed",
            "The workspace lock changed. Reload to continue.",
        ) from None


@router.post("/lock", response_model=WorkspaceLockStatusOut)
async def lock_workspace(request: Request) -> WorkspaceLockStatusOut:
    try:
        return _status_out(_lock(request).lock())
    except WorkspaceLockDisabled:
        raise api_error(
            409,
            "workspace-lock-disabled",
            "Turn on the workspace lock before locking.",
        ) from None


@router.get("/policy", response_model=WorkspaceLockPolicyOut)
def get_workspace_lock_policy(session: SessionDep) -> WorkspaceLockPolicyOut:
    try:
        return _policy_out(read_policy(session))
    except WorkspaceLockSettingInvalid:
        raise _setting_invalid() from None


def _chosen_pin(
    attempts: UnlockAttempts, saved: SavedWorkspaceLock, change: WorkspaceLockPolicyWrite
) -> PinVerifier | None:
    """The verifier a change saves, checking the current PIN first where it must.

    A change that replaces or removes the PIN, or turns the lock off, would
    otherwise open the workspace to anyone at the screen, so it goes through
    the same pacing as unlocking. Setting the first PIN needs nothing.
    """

    turns_off = saved.enabled and not change.enabled
    weakens = change.new_pin is not None or change.clear_pin or turns_off
    check_current = saved.pin is not None and weakens
    if not check_current and change.new_pin is None:
        return None if change.clear_pin else saved.pin
    wait = attempts.admit()
    if wait is not None:
        raise _throttled(wait)
    try:
        if check_current and saved.pin is not None:
            if not pin_matches(saved.pin, change.current_pin):
                raise _refused(attempts.fail())
            attempts.succeed()
        if change.new_pin is not None:
            return make_pin_verifier(change.new_pin)
        return None if change.clear_pin else saved.pin
    except PinCheckUnavailable:
        raise _unavailable() from None
    finally:
        attempts.release()


@router.put("/policy", response_model=WorkspaceLockPolicyOut)
def put_workspace_lock_policy(
    payload: WorkspaceLockPolicyWrite, request: Request, session: SessionDep
) -> WorkspaceLockPolicyOut:
    """Choose whether the workspace locks and which PIN opens it.

    A synchronous route, so nothing can come between the write and its commit;
    any derivation finishes before the write begins.
    """

    lock = _lock(request)
    try:
        saved = read_policy(session)
    except WorkspaceLockSettingInvalid:
        raise _setting_invalid() from None
    if saved.revision != payload.expected_revision:
        raise _setting_stale(saved.revision)
    if payload.new_pin is not None and not pin_within_bounds(payload.new_pin):
        raise api_error(422, "workspace-pin-invalid", "A PIN is 4 to 64 characters.")
    pin = _chosen_pin(lock.attempts, saved, payload)
    try:
        chosen = write_policy(
            session, expected_revision=payload.expected_revision, enabled=payload.enabled, pin=pin
        )
    except WorkspaceLockPolicyStale as stale:
        session.rollback()
        raise _setting_stale(stale.current_revision) from None
    session.commit()
    lock.note_policy(enabled=chosen.enabled, require_pin=chosen.pin is not None)
    return _policy_out(chosen)
