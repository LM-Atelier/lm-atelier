import { useMutation, type QueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { RecoveryCommand, RecoveryItem } from "./recoveryTypes";

export function useProjectDeletion(client: QueryClient) {
  const [deleted, setDeleted] = useState<RecoveryItem | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const commands = useRef(new Map<string, RecoveryCommand>());
  const refresh = () => {
    for (const key of ["projects", "chats", "chat-summaries", "chat", "chat-management", "recovery-items"])
      void client.invalidateQueries({ queryKey: [key] });
  };
  const undo = useMutation({
    mutationFn: async (item: RecoveryItem) => {
      let command = commands.current.get(item.deletion_id);
      if (!command) {
        const impact = await api.recoveryImpact(item.deletion_id);
        if (!impact.available_actions.includes("restore"))
          throw new ApiError(409, undefined, "This project cannot be restored now. Check Recently Deleted in Settings.", "recovery-restore-unavailable");
        command = { expected_revision: impact.revision, impact_sha256: impact.impact_sha256, operation_key: crypto.randomUUID() };
        commands.current.set(item.deletion_id, command);
      }
      return api.restoreRecovery(item.deletion_id, { ...command, restore_unfiled: false });
    },
    onSuccess: (result) => {
      commands.current.delete(result.deletion_id);
      setDeleted((current) => current?.deletion_id === result.deletion_id ? null : current);
      refresh();
    },
  });
  const mutation = useMutation({
    mutationFn: ({ id, command }: { id: string; command: RecoveryCommand }) => api.trashProject(id, command),
    onSuccess: (item) => {
      setDeleted(item); setNow(Date.now()); undo.reset(); refresh();
    },
  });
  useEffect(() => {
    if (!deleted) return;
    const deadline = new Date(deleted.purge_after).getTime();
    if (deadline <= now) return;
    const timer = window.setTimeout(() => setNow(Date.now()), Math.min(2_147_483_647, Math.max(0, deadline - Date.now())));
    return () => window.clearTimeout(timer);
  }, [deleted, now]);
  return { ...mutation, deleted, undo, expired: deleted !== null && new Date(deleted.purge_after).getTime() <= now,
    dismissUndo: () => { if (!undo.isPending) setDeleted(null); },
    recheckUndo: () => { if (!undo.isPending && deleted) { commands.current.delete(deleted.deletion_id); undo.reset(); } },
  };
}
