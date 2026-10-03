import { ApiError } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import type { useChatDeletion } from "./useChatDeletion";

export function ChatDeletionNotice({ deletion }: { deletion: ReturnType<typeof useChatDeletion> }) {
  const { deleted, undo, dismissUndo, recheckUndo } = deletion;
  if (!deleted) return null;
  const stale = undo.error instanceof ApiError && undo.error.status === 409;
  const expiry = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "long" }).format(new Date(deleted.item.purge_after));
  return <div className="toast recovery-notice" role="status">
    <div>
      <p>“{deleted.item.display_label}” moved to Recently Deleted.</p>
      <small>Recover until <time dateTime={deleted.item.purge_after}>{expiry}</time>.</small>
      <div className="row-actions">
        <button className="secondary compact-button" aria-disabled={undo.isPending || stale}
          onClick={() => { if (!undo.isPending && !stale) undo.mutate(deleted.item); }}>
          {undo.isPending ? "Restoring…" : undo.isError ? "Try Undo again" : "Undo"}
        </button>
        {stale && <button className="secondary compact-button" onClick={recheckUndo}>Recheck Undo</button>}
        <button className="secondary compact-button" aria-disabled={undo.isPending} onClick={dismissUndo}>Dismiss</button>
      </div>
      <ErrorCallout message={undo.error?.message} />
      {undo.isError && <p className="muted">You can also review this chat in Settings → Data & backups → Recently Deleted.</p>}
    </div>
  </div>;
}
