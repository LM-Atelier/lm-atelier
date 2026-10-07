import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { ApiError } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { MediaTrashConfirmation } from "./MediaTrashConfirmation";
import type { useMediaLibraryRecovery } from "./useMediaLibraryRecovery";

export function MediaLibraryRecovery({ recovery }: { recovery: ReturnType<typeof useMediaLibraryRecovery> }) {
  const { selected, deleted, undo, expired } = recovery;
  const undoButton = useRef<HTMLButtonElement>(null);
  useEffect(() => { if (deleted) undoButton.current?.focus(); }, [deleted]);
  const recheck = undo.error instanceof ApiError && undo.error.status >= 400 && undo.error.status < 500;
  return <>
    {selected && <MediaTrashConfirmation key={selected.id} entryId={selected.id} name={selected.display_name}
      onConfirm={recovery.confirm} onCancel={recovery.cancel} failure={recovery.trashError} onRecheck={recovery.recheckTrash} />}
    {deleted && createPortal(<div className="toast recovery-notice" role="status"><div>
      <p>“{deleted.display_label}” moved to Recently Deleted.</p>
      <small>{expired ? "Recovery ended" : "Recover until"} <time dateTime={deleted.purge_after}>
        {new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "long" }).format(new Date(deleted.purge_after))}
      </time>.</small>
      <div className="row-actions">
        <button ref={undoButton} className="secondary compact-button" aria-disabled={undo.isPending || recheck || expired}
          onClick={() => { if (!undo.isPending && !recheck && !expired) undo.mutate(deleted); }}>
          {undo.isPending ? "Restoring…" : undo.isError ? "Try Undo again" : "Undo"}
        </button>
        {recheck && !expired && <button className="secondary compact-button" onClick={recovery.recheckUndo}>Recheck Undo</button>}
        <button className="secondary compact-button" aria-disabled={undo.isPending} onClick={recovery.dismiss}>Dismiss</button>
      </div>
      <ErrorCallout message={undo.error?.message} />
      {undo.isError && <p className="muted">You can also review this item in Settings → Data & backups → Recently Deleted.</p>}
    </div></div>, document.querySelector(".notification-stack") ?? document.body)}
  </>;
}
