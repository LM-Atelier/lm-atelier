import { useEffect, useRef, useState } from "react";
import { ErrorCallout } from "./ErrorCallout";
import { WorkflowTrashConfirmation } from "./WorkflowTrashConfirmation";
import { useWorkflowRecovery } from "./useWorkflowRecovery";
import type { WorkflowFamily } from "./types";

export function WorkflowFamilyRecovery({ family, selectedId, onSelectionChange }: {
  family?: WorkflowFamily; selectedId: string | null; onSelectionChange: (id: string | null) => void;
}) {
  const [confirming, setConfirming] = useState<{ family: WorkflowFamily; workflowId: string } | null>(null);
  const recovery = useWorkflowRecovery(selectedId, onSelectionChange);
  const undoButton = useRef<HTMLButtonElement>(null);
  useEffect(() => { if (recovery.deleted && !confirming) undoButton.current?.focus(); }, [recovery.deleted, confirming]);
  const busy = recovery.undo.isPending || recovery.trash.isPending;
  return <>
    {family && selectedId && <div className="storage-actions"><button className="secondary"
      aria-disabled={busy} onClick={() => { if (!busy) setConfirming({ family, workflowId: selectedId }); }}>
      Delete workflow family
    </button></div>}
    {confirming && <WorkflowTrashConfirmation familyId={confirming.family.id} name={confirming.family.name}
      onCancel={() => setConfirming(null)} onConfirm={async (command) => {
        await recovery.trash.mutateAsync({ familyId: confirming.family.id, workflowId: confirming.workflowId,
          workflowIds: [confirming.workflowId, ...confirming.family.variants.map((variant) => variant.id)], command });
        setConfirming(null);
      }} />}
    {recovery.deleted && <div className="toast recovery-notice" role="status"><div>
      <p>“{recovery.deleted.item.display_label}” moved to Recently Deleted. Historical generation records stay available.</p>
      <small>{recovery.expired ? "Recovery ended" : "Recover until"} <time dateTime={recovery.deleted.item.purge_after}>
        {new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "long" }).format(new Date(recovery.deleted.item.purge_after))}
      </time>.</small>
      <p>Undo restores this family disabled. Review and enable it before using it again.</p>
      <div className="row-actions">
        <button ref={undoButton} className="secondary compact-button" aria-disabled={busy || recovery.stale || recovery.expired}
          onClick={recovery.restore}>{recovery.undo.isPending ? "Restoring…" : recovery.undo.isError ? "Try Undo again" : "Undo"}</button>
        {recovery.stale && !recovery.expired && <button className="secondary compact-button" onClick={recovery.recheck}>Recheck Undo</button>}
        <button className="secondary compact-button" aria-disabled={recovery.undo.isPending} onClick={recovery.dismiss}>Dismiss</button>
      </div>
      <ErrorCallout message={recovery.undo.error?.message} />
      {recovery.undo.isError && <p className="muted">You can also review this family in Settings → Data & backups → Recently Deleted.</p>}
    </div></div>}
  </>;
}
