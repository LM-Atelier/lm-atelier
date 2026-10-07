import { useEffect, useRef, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { LockedWorkspace } from "./LockedWorkspace";
import { useAppearance } from "./theme";
import { useWorkspaceUse } from "./useWorkspaceUse";
import { focusMainContent } from "./viewHelpers";
import { applyWorkspaceLockStatus, useWorkspaceLock, workspaceLockGeneration } from "./workspaceLockState";

/** How often a locked window asks whether it was unlocked somewhere else. */
const LOCKED_POLL_MS = 5_000;

/** Read the lock as the server has it now, and say whether an answer came back. */
async function readWorkspaceLock(): Promise<boolean> {
  const sentAt = workspaceLockGeneration();
  try {
    applyWorkspaceLockStatus(await api.workspaceLockStatus(), sentAt);
    return true;
  } catch {
    return false;
  }
}

function OpeningWorkspace() {
  // The workspace is not mounted yet, so nothing else has put the chosen theme on the page.
  useAppearance();
  return (
    <div className="first-run-shell">
      <p className="workspace-lock-pending" role="status">Opening LM Atelier…</p>
    </div>
  );
}

/** Shows the workspace only while the server says it is unlocked.
 *
 * Locking takes the whole workspace off the page rather than covering it, so
 * nothing it showed stays in the document, and its polling, live connection
 * and dialogs stop with it. What it had fetched is cleared from the cache, and
 * unlocking starts it again from nothing. The server refuses every request
 * while locked either way; this is what makes the page agree.
 *
 * If the lock cannot be read at all, the workspace is shown anyway: its own
 * requests then meet the server's refusal and bring the lock back here.
 */
export function WorkspaceLockGate({ children }: { children: ReactNode }) {
  const client = useQueryClient();
  const lock = useWorkspaceLock();
  useWorkspaceUse(lock.phase === "unlocked" && lock.enabled ? lock.idleSeconds : null);
  const [unreadable, setUnreadable] = useState(false);
  const blocked = lock.phase === "locked" || lock.phase === "changed";
  const wasBlocked = useRef(false);

  useEffect(() => {
    let active = true;
    void readWorkspaceLock().then((answered) => {
      if (active && !answered) setUnreadable(true);
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!blocked) return;
    wasBlocked.current = true;
    // Runs after the workspace has unmounted, so nothing is left to refetch
    // what is cleared, and an answer still in flight is dropped.
    void client.cancelQueries();
    client.clear();
    void readWorkspaceLock();
    const poll = window.setInterval(() => { void readWorkspaceLock(); }, LOCKED_POLL_MS);
    const onVisible = () => {
      if (document.visibilityState === "visible") void readWorkspaceLock();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearInterval(poll);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [blocked, client]);

  useEffect(() => {
    if (lock.phase !== "unlocked" || !wasBlocked.current) return;
    wasBlocked.current = false;
    focusMainContent();
  }, [lock.phase]);

  if (lock.phase === "locked" && lock.requirePin !== null) {
    return <LockedWorkspace requirePin={lock.requirePin} />;
  }
  if (lock.phase === "unlocked" || (lock.phase === "unknown" && unreadable)) return <>{children}</>;
  return <OpeningWorkspace />;
}
