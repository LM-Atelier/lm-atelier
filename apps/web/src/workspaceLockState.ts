import { useSyncExternalStore } from "react";
import type { WorkspaceLockStatus } from "./workspaceLockTypes";

/** What this window knows about the workspace lock.
 *
 * The server decides and enforces the lock; this only follows it, so the page
 * can take the workspace off screen, stop asking for data it will be refused,
 * and offer a way back in. It is a plain module rather than a query because
 * locking clears the query cache, and what remembers the lock must survive that.
 *
 * "changed" means the server has locked since this window last looked, even if
 * it is unlocked again now: what is on screen may predate the lock, so the
 * window starts over from a fresh read rather than carrying on.
 */
export type WorkspaceLockPhase = "unknown" | "unlocked" | "locked" | "changed";

export interface WorkspaceLockView {
  phase: WorkspaceLockPhase;
  enabled: boolean;
  /** Whether unlocking asks for a PIN, or null until the server has said. */
  requirePin: boolean | null;
  /** The lock epoch this window works under, sent with every request once known. */
  epoch: string | null;
  /** How long the workspace may go unused before it locks itself, or null for never. */
  idleSeconds: number | null;
}

const INITIAL: WorkspaceLockView = { phase: "unknown", enabled: false, requirePin: null, epoch: null, idleSeconds: null };

let view = INITIAL;
// Answers can arrive out of order: a status read sent before a refusal said
// the workspace locked may still come back saying it is unlocked. Every
// observation takes the next tick, and an answer sent before the latest
// observation pointing the other way is discarded.
let clock = 0;
let lockedAt = 0;
let unlockedAt = 0;
const listeners = new Set<() => void>();

function publish(next: WorkspaceLockView): void {
  if (
    next.phase === view.phase && next.enabled === view.enabled
    && next.requirePin === view.requirePin && next.epoch === view.epoch
    && next.idleSeconds === view.idleSeconds
  ) return;
  view = next;
  for (const listener of [...listeners]) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

/** The lock as this window knows it right now. */
export function workspaceLockView(): WorkspaceLockView {
  return view;
}

/** The tick to pass back with a status answer, read just before sending the request. */
export function workspaceLockGeneration(): number {
  return clock;
}

/** The server refused a request because the workspace is locked. */
export function noteWorkspaceLocked(): void {
  lockedAt = ++clock;
  // A PIN may have been set or removed since this window last asked, so a
  // fresh lock waits for the status read before offering a way back in.
  publish({ ...view, phase: "locked", requirePin: view.phase === "locked" ? view.requirePin : null });
}

/** The server refused a request carrying an epoch from before its latest lock. */
export function noteWorkspaceLockChanged(): void {
  lockedAt = ++clock;
  // Already locked is the stronger state: unlocking will start over anyway.
  if (view.phase !== "locked") publish({ ...view, phase: "changed" });
}

/** Take a status the server sent, unless something newer contradicts it.
 *
 * `sentAt` is the generation read just before the request went out. Without
 * it the status is an answer to this window's own lock or unlock and always
 * applies. Returns whether the status was taken.
 */
export function applyWorkspaceLockStatus(status: WorkspaceLockStatus, sentAt?: number): boolean {
  if (sentAt !== undefined && sentAt < (status.locked ? unlockedAt : lockedAt)) return false;
  clock += 1;
  if (status.locked) lockedAt = clock;
  else unlockedAt = clock;
  publish({
    phase: status.locked ? "locked" : "unlocked",
    enabled: status.enabled,
    requirePin: status.require_pin,
    epoch: status.lock_epoch,
    idleSeconds: typeof status.idle_lock_seconds === "number" ? status.idle_lock_seconds : null,
  });
  return true;
}

/** Follow what a new session says about the lock.
 *
 * An epoch is adopted only when this window holds none. A different one means
 * the workspace locked, or the service restarted, since this window last
 * looked; taking it quietly would let the page carry on as if nothing had
 * happened, so it is treated as a change instead. A server that says nothing
 * about the lock leaves everything as it was.
 */
export function noteSessionLock(session: { workspace_locked?: unknown; lock_epoch?: unknown }): void {
  if (typeof session.workspace_locked !== "boolean") return;
  const epoch = typeof session.lock_epoch === "string" && session.lock_epoch ? session.lock_epoch : null;
  if (epoch === null) publish({ ...view, epoch: null });
  else if (view.epoch === null) publish({ ...view, epoch });
  else if (view.epoch !== epoch) noteWorkspaceLockChanged();
  if (session.workspace_locked) noteWorkspaceLocked();
}

/** The epoch to send with a request, or null when none is known. */
export function workspaceLockEpoch(): string | null {
  return view.epoch;
}

/** Whether this window must not show or fetch workspace content right now. */
export function isWorkspaceLockBlocking(): boolean {
  return view.phase === "locked" || view.phase === "changed";
}

/** Whether an error is the server refusing a request because of the lock.
 *
 * Read from the error's shape rather than its class, so code that receives a
 * stand-in error, as tests do, still recognises it.
 */
export function isWorkspaceLockRefusal(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { status?: unknown }).status === 423;
}

/** What a failed lock, unlock or setting request said, read from its shape. */
export function workspaceLockFailure(error: unknown): { status: number | null; code: string | null; retryAfterSeconds: number } {
  const shape = typeof error === "object" && error !== null
    ? error as { status?: unknown; code?: unknown; payload?: unknown }
    : {};
  const payload = typeof shape.payload === "object" && shape.payload !== null
    ? shape.payload as { retry_after_seconds?: unknown }
    : {};
  const wait = payload.retry_after_seconds;
  return {
    status: typeof shape.status === "number" ? shape.status : null,
    code: typeof shape.code === "string" ? shape.code : null,
    retryAfterSeconds: typeof wait === "number" && Number.isFinite(wait) && wait > 0 ? Math.ceil(wait) : 0,
  };
}

/** The lock as this window knows it, re-rendering whoever reads it when that changes. */
export function useWorkspaceLock(): WorkspaceLockView {
  return useSyncExternalStore(subscribe, workspaceLockView, workspaceLockView);
}

/** Forget everything, so each test starts from a window that has not asked yet. */
export function resetWorkspaceLockForTests(): void {
  clock = 0;
  lockedAt = 0;
  unlockedAt = 0;
  publish(INITIAL);
}
