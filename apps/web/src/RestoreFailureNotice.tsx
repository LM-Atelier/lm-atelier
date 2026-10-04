import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { BackupRestoreState } from "./types";

type Reason = NonNullable<BackupRestoreState["reason"]>;

// In this page's own words, so the notice never depends on server text.
const REASONS: Record<Reason, string> = {
  "backup-missing": "The backup it was asked to use was no longer there.",
  "backup-invalid": "The backup did not pass its checks.",
  "backup-newer": "The backup was made by a newer version of LM Atelier.",
  "restore-failed": "The backup could not be put in place.",
};

/** Says that a restore asked for could not be applied at the last start, until it is dismissed.
 *
 * The data was left exactly as it was, and LM Atelier started with it. Asking
 * for a restore again replaces the notice; so does dismissing it.
 */
export function RestoreFailureNotice() {
  const client = useQueryClient();
  const restore = useQuery({ queryKey: ["backups", "restore-state"], queryFn: () => api.backupRestoreState() });
  const dismiss = useMutation({
    mutationFn: () => api.dismissFailedRestore(),
    onSuccess: () => client.invalidateQueries({ queryKey: ["backups", "restore-state"] }),
  });
  const state = restore.data;
  if (!state || state.state !== "failed") return null;
  return (
    <div className="callout warning action-callout" role="status">
      <span>
        The restore asked for was not applied when LM Atelier last started, and your data was left as it was.{" "}
        {REASONS[state.reason ?? "restore-failed"]}
      </span>
      <button
        type="button"
        className="secondary compact-button"
        aria-disabled={dismiss.isPending}
        onClick={() => {
          if (!dismiss.isPending) dismiss.mutate();
        }}
      >
        Dismiss
      </button>
      {dismiss.isError && <span role="alert">The notice could not be dismissed. Try again.</span>}
    </div>
  );
}
