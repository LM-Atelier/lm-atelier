import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  applyWorkspaceLockStatus,
  isWorkspaceLockBlocking,
  isWorkspaceLockRefusal,
  noteSessionLock,
  noteWorkspaceLockChanged,
  noteWorkspaceLocked,
  resetWorkspaceLockForTests,
  useWorkspaceLock,
  workspaceLockEpoch,
  workspaceLockFailure,
  workspaceLockGeneration,
  workspaceLockView,
} from "./workspaceLockState";

const unlocked = { locked: false, enabled: true, require_pin: false, lock_epoch: "epoch-one" };
const locked = { locked: true, enabled: true, require_pin: true, lock_epoch: "epoch-two" };

beforeEach(() => { resetWorkspaceLockForTests(); });
afterEach(() => {
  cleanup();
  resetWorkspaceLockForTests();
});

describe("answers that arrive out of order", () => {
  it("ignores an unlocked answer sent before a refusal said the workspace locked", () => {
    const sentAt = workspaceLockGeneration();
    noteWorkspaceLocked();

    expect(applyWorkspaceLockStatus(unlocked, sentAt)).toBe(false);
    expect(isWorkspaceLockBlocking()).toBe(true);
    expect(workspaceLockEpoch()).toBeNull();
  });

  it("takes an unlocked answer sent after the latest lock it saw", () => {
    noteWorkspaceLocked();
    const sentAt = workspaceLockGeneration();

    expect(applyWorkspaceLockStatus(unlocked, sentAt)).toBe(true);
    expect(isWorkspaceLockBlocking()).toBe(false);
    expect(workspaceLockEpoch()).toBe("epoch-one");
  });

  it("does not let a locked answer sent before this window's own unlock lock it again", () => {
    applyWorkspaceLockStatus(locked);
    const sentAt = workspaceLockGeneration();
    applyWorkspaceLockStatus(unlocked);

    expect(applyWorkspaceLockStatus(locked, sentAt)).toBe(false);
    expect(isWorkspaceLockBlocking()).toBe(false);
  });

  it("always takes a locked answer when nothing unlocked this window since it was sent", () => {
    const sentAt = workspaceLockGeneration();
    noteSessionLock({ workspace_locked: true, lock_epoch: "epoch-two" });

    expect(applyWorkspaceLockStatus(locked, sentAt)).toBe(true);
    const { result } = renderHook(() => useWorkspaceLock());
    expect(result.current).toEqual({ phase: "locked", enabled: true, requirePin: true, epoch: "epoch-two", idleSeconds: null });
  });

  it("treats a superseded epoch as a change, and a lock as stronger than a change", () => {
    applyWorkspaceLockStatus(unlocked);
    noteWorkspaceLockChanged();
    const { result } = renderHook(() => useWorkspaceLock());
    expect(result.current.phase).toBe("changed");

    act(() => noteWorkspaceLocked());
    expect(result.current.phase).toBe("locked");
    act(() => noteWorkspaceLockChanged());
    expect(result.current.phase).toBe("locked");
  });

  it("asks again whether unlocking needs a PIN when a refusal is what says it locked", () => {
    applyWorkspaceLockStatus(unlocked);
    noteWorkspaceLocked();
    expect(workspaceLockView().requirePin).toBeNull();

    expect(applyWorkspaceLockStatus(locked, workspaceLockGeneration())).toBe(true);
    noteWorkspaceLocked();
    expect(workspaceLockView().requirePin).toBe(true);
  });
});

describe("what a new session says", () => {
  it("adopts an epoch when this window holds none", () => {
    noteSessionLock({ workspace_locked: false, lock_epoch: "epoch-one" });

    expect(workspaceLockEpoch()).toBe("epoch-one");
    expect(isWorkspaceLockBlocking()).toBe(false);
  });

  it("flags a different epoch as a change instead of adopting it", () => {
    noteSessionLock({ workspace_locked: false, lock_epoch: "epoch-one" });
    noteSessionLock({ workspace_locked: false, lock_epoch: "epoch-two" });

    expect(workspaceLockEpoch()).toBe("epoch-one");
    const { result } = renderHook(() => useWorkspaceLock());
    expect(result.current.phase).toBe("changed");
  });

  it("clears the epoch when the lock is off", () => {
    noteSessionLock({ workspace_locked: false, lock_epoch: "epoch-one" });
    noteSessionLock({ workspace_locked: false, lock_epoch: null });

    expect(workspaceLockEpoch()).toBeNull();
    expect(isWorkspaceLockBlocking()).toBe(false);
  });

  it("locks when the session says the workspace is locked", () => {
    noteSessionLock({ workspace_locked: true, lock_epoch: "epoch-one" });

    expect(isWorkspaceLockBlocking()).toBe(true);
  });

  it("changes nothing when the server says nothing about the lock", () => {
    noteSessionLock({ workspace_locked: false, lock_epoch: "epoch-one" });
    noteSessionLock({});

    expect(workspaceLockEpoch()).toBe("epoch-one");
    expect(isWorkspaceLockBlocking()).toBe(false);
  });
});

describe("reading refusals", () => {
  it("recognises a lock refusal by its status alone", () => {
    expect(isWorkspaceLockRefusal({ status: 423, code: "workspace-locked" })).toBe(true);
    expect(isWorkspaceLockRefusal({ status: 403, code: "csrf-invalid" })).toBe(false);
    expect(isWorkspaceLockRefusal(new Error("neutral failure"))).toBe(false);
    expect(isWorkspaceLockRefusal(null)).toBe(false);
  });

  it("reads the wait a refusal names, and nothing from anything else", () => {
    expect(workspaceLockFailure({ status: 429, code: "workspace-pin-throttled", payload: { retry_after_seconds: 2.2 } }))
      .toEqual({ status: 429, code: "workspace-pin-throttled", retryAfterSeconds: 3 });
    expect(workspaceLockFailure({ status: 403, code: "workspace-pin-refused", payload: { retry_after_seconds: 0 } }).retryAfterSeconds).toBe(0);
    expect(workspaceLockFailure(new Error("neutral failure"))).toEqual({ status: null, code: null, retryAfterSeconds: 0 });
  });
});
