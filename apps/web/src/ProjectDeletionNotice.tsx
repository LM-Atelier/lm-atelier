import { useEffect, useRef } from "react";
import { ApiError } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import type { useProjectDeletion } from "./useProjectDeletion";

export function ProjectDeletionNotice({ deletion }: { deletion: ReturnType<typeof useProjectDeletion> }) {
  const { deleted, undo, expired, dismissUndo, recheckUndo } = deletion;
  const undoButton = useRef<HTMLButtonElement>(null);
  useEffect(() => { if (deleted) undoButton.current?.focus(); }, [deleted]);
  if (!deleted) return null;
  const stale = undo.error instanceof ApiError && undo.error.status >= 400 && undo.error.status < 500;
  return <div className="toast recovery-notice" role="status"><div>
    <p>“{deleted.display_label}” moved to Recently Deleted. Its chats stay available.</p>
    <small>{expired ? "Recovery ended" : "Recover until"} <time dateTime={deleted.purge_after}>
      {new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "long" }).format(new Date(deleted.purge_after))}
    </time>.</small>
    <div className="row-actions">
      <button ref={undoButton} className="secondary compact-button" aria-disabled={undo.isPending || stale || expired}
        onClick={() => { if (!undo.isPending && !stale && !expired) undo.mutate(deleted); }}>
        {undo.isPending ? "Restoring…" : undo.isError ? "Try Undo again" : "Undo"}
      </button>
      {stale && !expired && <button className="secondary compact-button" onClick={recheckUndo}>Recheck Undo</button>}
      <button className="secondary compact-button" aria-disabled={undo.isPending} onClick={dismissUndo}>Dismiss</button>
    </div>
    <ErrorCallout message={undo.error?.message} />
    {undo.isError && <p className="muted">You can also review this project in Settings → Data & backups → Recently Deleted.</p>}
  </div></div>;
}
