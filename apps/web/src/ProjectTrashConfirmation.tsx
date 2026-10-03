import { useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api, ApiError } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
import { ErrorCallout } from "./ErrorCallout";
import type { RecoveryCommand } from "./recoveryTypes";

export function ProjectTrashConfirmation({ projectId, name, onConfirm, onCancel }: {
  projectId: string; name: string;
  onConfirm: (command: RecoveryCommand) => void | Promise<void>;
  onCancel: () => void;
}) {
  const [operationKey, setOperationKey] = useState(() => crypto.randomUUID());
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const command = useRef<RecoveryCommand | null>(null);
  const submitting = useRef(false);
  const impact = useQuery({
    queryKey: ["project-deletion-impact", projectId, operationKey],
    queryFn: ({ signal }) => api.projectDeletionImpact(projectId, signal),
    retry: false, refetchOnWindowFocus: false, refetchOnReconnect: false,
  });
  const allowed = impact.data?.available_actions.includes("trash") === true;
  const recheck = error instanceof ApiError && error.status >= 400 && error.status < 500;
  const ready = allowed && !impact.isFetching && !impact.isError && !pending && !recheck;
  return <ConfirmDialog title="Move this project to Recently Deleted?"
    question={`Move “${name}” to Recently Deleted? You can restore its settings for 30 days. Its chats stay available as unfiled.`}
    tone="action" confirmLabel={pending ? "Moving project…" : error && !recheck ? "Try again" : "Move to Recently Deleted"}
    confirmDisabled={!ready} onCancel={() => { if (!submitting.current) onCancel(); }}
    onConfirm={async () => {
      if (!ready || !impact.data || submitting.current) return;
      submitting.current = true; setPending(true); setError(null);
      command.current ??= { expected_revision: impact.data.revision, impact_sha256: impact.data.impact_sha256, operation_key: operationKey };
      try { await onConfirm(command.current); }
      catch (failure) { setError(failure instanceof Error ? failure : new Error("This project could not be moved. Try again.")); }
      finally { submitting.current = false; setPending(false); }
    }} detail={<>
      {impact.isFetching && <p role="status">Checking this project's current deletion details…</p>}
      {impact.data && <p>{impact.data.counts.chats} chats stay available. No chat history or media is deleted.</p>}
      <p>New work stops inheriting this project's defaults. Already accepted work keeps its settings. Restoring does not restart work or move chats filed elsewhere.</p>
      {impact.data && !allowed && <p role="alert">This project cannot be moved now. Check its current state again.</p>}
      <ErrorCallout message={error?.message ?? impact.error?.message} />
      {(recheck || impact.isError || !allowed) && !impact.isFetching && !pending && <button className="secondary"
        onClick={() => { command.current = null; setError(null); setOperationKey(crypto.randomUUID()); }}>Check again</button>}
    </>} />;
}
