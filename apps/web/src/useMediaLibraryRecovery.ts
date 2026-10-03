import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { ArtifactLibraryEntry } from "./artifactLibraryPage";
import type { RecoveryCommand, RecoveryItem } from "./recoveryTypes";

export function useMediaLibraryRecovery() {
  const client = useQueryClient();
  const [selected, setSelected] = useState<ArtifactLibraryEntry | null>(null);
  const [deleted, setDeleted] = useState<RecoveryItem | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const undoCommand = useRef<{ deletionId: string; command: RecoveryCommand } | null>(null);
  const refresh = () => {
    for (const key of ["artifact-library-v1", "artifacts", "artifact-storage", "recovery-items"])
      void client.invalidateQueries({ queryKey: [key] });
  };
  const undo = useMutation({
    mutationFn: async (item: RecoveryItem) => {
      if (undoCommand.current?.deletionId !== item.deletion_id) {
        const impact = await api.recoveryImpact(item.deletion_id);
        if (!impact.available_actions.includes("restore"))
          throw new ApiError(409, undefined, "This item cannot be restored now. Check Recently Deleted in Settings.", "recovery-restore-unavailable");
        undoCommand.current = { deletionId: item.deletion_id, command: {
          expected_revision: impact.revision, impact_sha256: impact.impact_sha256, operation_key: crypto.randomUUID(),
        } };
      }
      return api.restoreRecovery(item.deletion_id, { ...undoCommand.current.command, restore_unfiled: false });
    },
    onSuccess: (result) => {
      setDeleted((current) => current?.deletion_id === result.deletion_id ? null : current);
      refresh();
    },
  });
  const trash = useMutation({
    mutationFn: ({ entry, command }: { entry: ArtifactLibraryEntry; command: RecoveryCommand }) => api.trashMedia(entry.id, command),
    onSuccess: (item) => {
      setSelected(null);
      setDeleted(item);
      setNow(Date.now());
      undoCommand.current = null;
      undo.reset();
      refresh();
    },
  });
  useEffect(() => {
    if (!deleted) return;
    const deadline = new Date(deleted.purge_after).getTime();
    if (deadline <= now) return;
    const timer = window.setTimeout(() => setNow(Date.now()), Math.min(2_147_483_647, Math.max(0, deadline - Date.now())));
    return () => window.clearTimeout(timer);
  }, [deleted, now]);
  const busy = trash.isPending || undo.isPending;
  return {
    selected, deleted, undo, busy, trashError: trash.error,
    expired: deleted !== null && new Date(deleted.purge_after).getTime() <= now,
    choose: (entry: ArtifactLibraryEntry) => { if (!busy) { trash.reset(); setSelected(entry); } },
    recheckTrash: () => { if (!trash.isPending) trash.reset(); },
    cancel: () => { if (!busy) setSelected(null); },
    confirm: async (command: RecoveryCommand) => { if (selected) await trash.mutateAsync({ entry: selected, command }); },
    dismiss: () => { if (!undo.isPending) setDeleted(null); },
    recheckUndo: () => { if (!undo.isPending) { undoCommand.current = null; undo.reset(); } },
  };
}
