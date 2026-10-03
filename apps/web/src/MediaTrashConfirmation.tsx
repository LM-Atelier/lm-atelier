import { useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api, ApiError } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";
import type { RecoveryCommand } from "./recoveryTypes";

export function MediaTrashConfirmation({ entryId, name, onConfirm, onCancel, failure, onRecheck }: {
  entryId: string; name: string;
  onConfirm: (command: RecoveryCommand) => Promise<void>;
  onCancel: () => void;
  failure?: Error | null;
  onRecheck?: () => void;
}) {
  const [operationKey] = useState(() => crypto.randomUUID());
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const submitting = useRef(false);
  const impact = useQuery({
    queryKey: ["media-deletion-impact", entryId, operationKey],
    queryFn: ({ signal }) => api.mediaDeletionImpact(entryId, signal),
    retry: false, refetchOnWindowFocus: false, refetchOnReconnect: false,
  });
  const allowed = impact.data?.available_actions.includes("trash") === true;
  const currentError = error ?? failure;
  const recheck = currentError instanceof ApiError && currentError.status >= 400 && currentError.status < 500;
  const ready = allowed && !impact.isFetching && !impact.isError && !pending && !recheck;
  return <ConfirmDialog title="Move this Media Library item to Recently Deleted?"
    question={`Move “${name}” to Recently Deleted? You can restore it with its favorites, collections and tags for 30 days.`}
    tone="action" confirmLabel={pending ? "Moving item…" : "Move to Recently Deleted"} confirmDisabled={!ready}
    onCancel={() => { if (!submitting.current) onCancel(); }} onConfirm={async () => {
      if (!ready || !impact.data || submitting.current) return;
      submitting.current = true; setPending(true); setError(null);
      try {
        await onConfirm({ expected_revision: impact.data.revision, impact_sha256: impact.data.impact_sha256, operation_key: operationKey });
      } catch (failure) {
        setError(failure instanceof Error ? failure : new Error("This item could not be moved. Try again."));
      } finally { submitting.current = false; setPending(false); }
    }} detail={<>
      {impact.isFetching && <p role="status">Checking this item's current deletion details…</p>}
      {impact.data && <p>{impact.data.counts.artifacts} media files · Retained media: {formatBytes(impact.data.counts.retained_bytes)} (estimate)</p>}
      <p>No media bytes are removed now. Existing chats and References keep their media.</p>
      {impact.data && !allowed && <p role="alert">This item cannot be moved now. Check its current state again.</p>}
      <ErrorCallout message={currentError?.message ?? impact.error?.message} />
      {(recheck || impact.isError || !allowed) && !impact.isFetching && !pending && <button className="secondary"
        onClick={() => { setError(null); onRecheck?.(); void impact.refetch(); }}>Check again</button>}
    </>} />;
}
