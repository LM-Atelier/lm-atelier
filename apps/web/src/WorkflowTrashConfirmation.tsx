import { useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api, ApiError } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
import { ErrorCallout } from "./ErrorCallout";
import type { RecoveryCommand } from "./recoveryTypes";

export function WorkflowTrashConfirmation({ familyId, name, onConfirm, onCancel }: {
  familyId: string; name: string;
  onConfirm: (command: RecoveryCommand) => Promise<void>;
  onCancel: () => void;
}) {
  const [operationKey, setOperationKey] = useState(() => crypto.randomUUID());
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const command = useRef<RecoveryCommand | null>(null);
  const submitting = useRef(false);
  const impact = useQuery({
    queryKey: ["workflow-deletion-impact", familyId, operationKey],
    queryFn: ({ signal }) => api.workflowDeletionImpact(familyId, signal),
    retry: false, refetchOnWindowFocus: false, refetchOnReconnect: false,
  });
  const allowed = impact.data?.available_actions.includes("trash") === true;
  const recheck = error instanceof ApiError && error.status >= 400 && error.status < 500;
  const ready = allowed && !impact.isFetching && !impact.isError && !pending && !recheck;
  return <ConfirmDialog title="Move this workflow family to Recently Deleted?"
    question={`Move “${name}” and all its variants to Recently Deleted? You can restore them for 30 days.`}
    tone="action" confirmLabel={pending ? "Moving workflow…" : error && !recheck ? "Try again" : "Move to Recently Deleted"}
    confirmDisabled={!ready} onCancel={() => { if (!submitting.current) onCancel(); }}
    onConfirm={async () => {
      if (!ready || !impact.data || submitting.current) return;
      submitting.current = true; setPending(true); setError(null);
      command.current ??= { expected_revision: impact.data.revision, impact_sha256: impact.data.impact_sha256, operation_key: operationKey };
      try { await onConfirm(command.current); }
      catch (failure) { setError(failure instanceof Error ? failure : new Error("This workflow could not be moved. Try again.")); }
      finally { submitting.current = false; setPending(false); }
    }} detail={<>
      {impact.isFetching && <p role="status">Checking this workflow's current deletion details…</p>}
      {impact.data && <p>{impact.data.counts.workflow_definitions} variants and {impact.data.counts.workflow_revisions} revisions move together. Historical generation records stay available.</p>}
      <p>Restoring keeps this family disabled and non-default. Review and enable it before using it again. Undo does not trust revisions or activate workers.</p>
      {impact.data?.conflicts.includes("active_selection") && <p role="alert">This family is selected by a chat or project, or is set as a default. Change those selections before deleting it.</p>}
      {impact.data?.conflicts.includes("active_work") && <p role="alert">Queued, held or running work still needs this family. Let that work finish or cancel it before deleting the family.</p>}
      {impact.data && !allowed && !impact.data.conflicts.includes("active_selection") && !impact.data.conflicts.includes("active_work") &&
        <p role="alert">This workflow cannot be moved now. Check its current state again.</p>}
      <ErrorCallout message={error?.message ?? impact.error?.message} />
      {(recheck || impact.isError || !allowed) && !impact.isFetching && !pending && <button className="secondary"
        onClick={() => { command.current = null; setError(null); setOperationKey(crypto.randomUUID()); }}>Check again</button>}
    </>} />;
}
