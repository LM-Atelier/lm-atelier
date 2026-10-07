import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";

/** Says that a backup from the list will replace the current data at the next start, and withdraws it on request.
 *
 * Until LM Atelier restarts nothing has changed, so cancelling keeps the
 * current data and the backup itself.
 */
export function ScheduledRestoreNotice({ onCancelled }: { onCancelled: () => void }) {
  const client = useQueryClient();
  const cancel = useMutation({
    mutationFn: () => api.cancelRestore(),
    onSuccess: async () => {
      onCancelled();
      await client.invalidateQueries({ queryKey: ["backups"] });
    },
  });
  return (
    <div className="callout success action-callout" role="status">
      <span>Restore scheduled. Restart LM Atelier to apply the selected backup.</span>
      <button
        type="button"
        className="secondary compact-button"
        aria-disabled={cancel.isPending}
        onClick={() => {
          if (!cancel.isPending) cancel.mutate();
        }}
      >
        {cancel.isPending ? "Cancelling…" : "Cancel restore"}
      </button>
      {cancel.isError && <span role="alert">The restore could not be cancelled. Try again.</span>}
    </div>
  );
}
