import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
import { formatBytes } from "./format";
import type { ArtifactCleanupResult, ArtifactStorageInfo } from "./types";

type Cleared = { removed: number; reclaimed: number };

function count(value: number, one: string, many: string): string {
  return `${value.toLocaleString()} ${value === 1 ? one : many}`;
}

/** Clear, now, the media retention would clear on its own at the next start.
 *
 * Retention runs once when the workspace starts, so whatever becomes ready
 * afterwards waits for a restart. This asks the server for a fresh preview,
 * says what it found and by which rule, and only then runs the same retention
 * the server runs at startup. Nothing is chosen here: the server decides again
 * at deletion time what is still unused, so a file that came back into use in
 * the meantime is kept.
 */
export function ClearReadyMedia({ info }: { info: ArtifactStorageInfo }) {
  const client = useQueryClient();
  const [preview, setPreview] = useState<ArtifactCleanupResult | null>(null);
  const [outcome, setOutcome] = useState<{ kind: "success" | "error"; message: string } | null>(null);
  const refresh = () => void client.invalidateQueries({ queryKey: ["artifact-storage"] });

  const check = useMutation({
    mutationFn: () => api.cleanupArtifacts(true),
    onMutate: () => setOutcome(null),
    onSuccess: (found) => {
      if (found.removed_count > 0) {
        setPreview(found);
        return;
      }
      setOutcome({ kind: "success", message: "Nothing is ready to clear any more." });
      refresh();
    },
    onError: (error) => setOutcome({ kind: "error", message: error.message }),
  });

  const clear = useMutation({
    mutationFn: async (): Promise<Cleared> => {
      const cleared: Cleared = { removed: 0, reclaimed: 0 };
      // A real run is one bounded batch, and a truncated one leaves the rest
      // for the next call. A truncated batch that removed nothing would ask
      // again forever, so that ends the run as well.
      for (;;) {
        const batch = await api.cleanupArtifacts(false);
        cleared.removed += batch.removed_count;
        cleared.reclaimed += batch.reclaimed_bytes;
        if (!batch.truncated || batch.removed_count === 0) return cleared;
      }
    },
    onSuccess: ({ removed, reclaimed }) => {
      setOutcome({
        kind: "success",
        message: removed > 0 ? `Cleared ${count(removed, "file", "files")}, ${formatBytes(reclaimed)}.` : "Nothing needed clearing after all.",
      });
    },
    onError: (error) => {
      // Every batch before the failure was committed on its own.
      setOutcome({
        kind: "error",
        message: `Clearing stopped before it finished, and anything already cleared stays cleared. ${error.message}`,
      });
    },
    onSettled: () => {
      setPreview(null);
      refresh();
    },
  });

  return (
    <>
      {info.eligible_count > 0 && (
        <div className="row-actions">
          <button
            type="button"
            className="secondary"
            aria-disabled={check.isPending || clear.isPending}
            onClick={() => {
              if (check.isPending || clear.isPending) return;
              check.mutate();
            }}
          >
            {check.isPending ? "Checking…" : "Clear now"}
          </button>
        </div>
      )}
      {outcome && (
        <div className={`callout ${outcome.kind}`} role={outcome.kind === "error" ? "alert" : "status"}>
          {outcome.message}
        </div>
      )}
      {preview && (
        <ConfirmDialog
          title="Clear media nothing uses?"
          question={`${count(preview.removed_count, "file", "files")}, ${formatBytes(preview.reclaimed_bytes)}.`}
          detail={
            <p>
              {"Only files that no chat, Media Library item, reference or past run uses, and that are not favorites: "}
              {`previews and in-between steps once they are ${count(info.temporary_retention_hours, "hour", "hours")} old, `}
              {`and anything else once it has gone ${count(info.retention_days, "day", "days")} unused. `}
              They cannot be restored.
            </p>
          }
          confirmLabel={clear.isPending ? "Clearing…" : "Clear"}
          confirmDisabled={clear.isPending}
          onConfirm={() => clear.mutate()}
          onCancel={() => {
            if (!clear.isPending) setPreview(null);
          }}
        />
      )}
    </>
  );
}
