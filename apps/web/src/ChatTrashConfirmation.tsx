import { useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api, ApiError } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";
import type { RecoveryCommand } from "./recoveryTypes";

export function ChatTrashConfirmation({ chatId, title, deleteGeneratedMedia, onConfirm, onCancel }: {
  chatId: string;
  title: string;
  deleteGeneratedMedia: boolean;
  onConfirm: (command: RecoveryCommand) => void | Promise<void>;
  onCancel: () => void;
}) {
  const [operationKey] = useState(() => crypto.randomUUID());
  const [pending, setPending] = useState(false);
  const submitting = useRef(false);
  const [error, setError] = useState<Error | null>(null);
  const impact = useQuery({
    queryKey: ["chat-deletion-impact", chatId, deleteGeneratedMedia, operationKey],
    queryFn: ({ signal }) => api.deletionImpact(chatId, deleteGeneratedMedia, signal),
    retry: false, refetchOnWindowFocus: false, refetchOnReconnect: false,
  });
  const allowed = impact.data?.available_actions.includes("trash") === true;
  const requiresRecheck = error instanceof ApiError && error.status >= 400 && error.status < 500;
  const ready = allowed && !impact.isFetching && !impact.isError && !pending && !requiresRecheck;
  return <ConfirmDialog title="Move this chat to Recently Deleted?"
    question={`Move “${title}” and its history to Recently Deleted? You can restore it for 30 days.`}
    tone="action" confirmLabel={pending ? "Moving chat…" : "Move to Recently Deleted"} confirmDisabled={!ready}
    onCancel={() => { if (!submitting.current) onCancel(); }} onConfirm={async () => {
      if (!ready || !impact.data || submitting.current) return;
      submitting.current = true;
      setPending(true);
      setError(null);
      try {
        await onConfirm({ expected_revision: impact.data.revision, impact_sha256: impact.data.impact_sha256, operation_key: operationKey });
      } catch (failure) {
        setError(failure instanceof Error ? failure : new Error("This chat could not be moved. Try again."));
      } finally {
        submitting.current = false;
        setPending(false);
      }
    }} detail={<>
      {impact.isFetching && <p role="status">Checking this chat's current deletion details…</p>}
      {impact.data && <p>
        {impact.data.counts.messages} messages · {impact.data.counts.artifacts} media files
        <br />Retained media: {formatBytes(impact.data.counts.retained_bytes)} (estimate)
        <br />No media bytes are removed now.
      </p>}
      {deleteGeneratedMedia && <p>At permanent deletion, exclusive generated media also leaves the Media Library. Favorites and media retained elsewhere stay protected.</p>}
      {impact.data && !allowed && <p role="alert">Wait for this chat's work to finish before deleting it.</p>}
      <ErrorCallout message={error?.message ?? impact.error?.message} />
      {(requiresRecheck || impact.isError || !allowed) && !impact.isFetching && !pending && <button className="secondary" onClick={() => { setError(null); void impact.refetch(); }}>Check again</button>}
    </>} />;
}
