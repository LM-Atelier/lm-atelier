import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { RecoveryCommand, RecoveryItem } from "./recoveryTypes";

type DeletedWorkflow = { item: RecoveryItem; workflowId: string };

export function useWorkflowRecovery(selectedId: string | null, onSelectionChange: (id: string | null) => void) {
  const client = useQueryClient();
  const [deleted, setDeleted] = useState<DeletedWorkflow | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const current = useRef({ selectedId, deleted, onSelectionChange });
  const commands = useRef(new Map<string, RecoveryCommand>());
  const restoring = useRef(false);
  useEffect(() => { current.current = { selectedId, deleted, onSelectionChange }; }, [selectedId, deleted, onSelectionChange]);
  const refresh = () => {
    for (const key of ["workflow-families", "workflow-family", "workflows", "workflow-revision", "workflow-ready-revisions", "studio-capabilities", "recovery-items"])
      void client.invalidateQueries({ queryKey: [key] });
  };
  const undo = useMutation({
    mutationFn: async ({ item }: DeletedWorkflow) => {
      let command = commands.current.get(item.deletion_id);
      if (!command) {
        const impact = await api.recoveryImpact(item.deletion_id);
        if (!impact.available_actions.includes("restore"))
          throw new ApiError(409, undefined, "This workflow cannot be restored now. Check Recently Deleted in Settings.", "recovery-restore-unavailable");
        command = { expected_revision: impact.revision, impact_sha256: impact.impact_sha256, operation_key: crypto.randomUUID() };
        commands.current.set(item.deletion_id, command);
      }
      return api.restoreRecovery(item.deletion_id, { ...command, restore_unfiled: false });
    },
    onSuccess: (result, target) => {
      commands.current.delete(result.deletion_id);
      if (current.current.deleted?.item.deletion_id === result.deletion_id && current.current.selectedId === null)
        current.current.onSelectionChange(target.workflowId);
      setDeleted((value) => value?.item.deletion_id === result.deletion_id ? null : value);
      refresh();
    },
    onSettled: () => { restoring.current = false; },
  });
  const trash = useMutation({
    mutationFn: ({ familyId, command }: { familyId: string; workflowId: string; workflowIds: string[]; command: RecoveryCommand }) => api.trashWorkflow(familyId, command),
    onSuccess: (item, target) => {
      setDeleted({ item, workflowId: target.workflowId }); setNow(Date.now()); undo.reset(); refresh();
      if (current.current.selectedId !== null && target.workflowIds.includes(current.current.selectedId)) current.current.onSelectionChange(null);
    },
  });
  useEffect(() => {
    if (!deleted) return;
    const deadline = new Date(deleted.item.purge_after).getTime();
    if (deadline <= now) return;
    const timer = window.setTimeout(() => setNow(Date.now()), Math.min(2_147_483_647, Math.max(0, deadline - Date.now())));
    return () => window.clearTimeout(timer);
  }, [deleted, now]);
  const expired = deleted !== null && new Date(deleted.item.purge_after).getTime() <= now;
  const stale = undo.error !== null && undo.error instanceof ApiError && undo.error.status >= 400 && undo.error.status < 500;
  return { trash, deleted, undo, expired, stale,
    restore: () => {
      if (!deleted || expired || stale || restoring.current || trash.isPending) return;
      restoring.current = true; undo.mutate(deleted);
    },
    dismiss: () => { if (!restoring.current) setDeleted(null); },
    recheck: () => { if (!restoring.current && deleted) { commands.current.delete(deleted.item.deletion_id); undo.reset(); } },
  };
}
