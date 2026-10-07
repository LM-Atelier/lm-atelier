import { useState } from "react";
import { GitCommitVertical } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { artifactSource, type EditLineageStep } from "./messageMedia";

/** Show every edit that led to a result, oldest first.
 *
 * The compare slider answers "what changed in this step"; this answers "how
 * did we get here" once a result is at least two edits deep. Each entry shows
 * the image entering the step and the exact instruction that transformed it,
 * ending with the current result.
 */
export function LineageButton({
  steps,
  resultUrl,
  paging,
}: {
  steps: EditLineageStep[];
  resultUrl: string;
  paging?: { hasOlder: boolean; loading: boolean; error: boolean; onOlder: () => void };
}) {
  const [open, setOpen] = useState(false);
  if (steps.length < 2 && !paging?.hasOlder) return null;
  return (
    <>
      <button
        type="button"
        className="icon-button"
        aria-label="Show the edit lineage"
        title="Lineage"
        onClick={() => setOpen(true)}
      >
        <GitCommitVertical size={14} aria-hidden="true" />
      </button>
      {open && (
        <AccessibleDialog
          title="Edit lineage"
          eyebrow={`${paging?.hasOlder ? "Latest " : ""}${steps.length} steps`}
          closeLabel="Close lineage"
          onClose={() => setOpen(false)}
          className="lineage-dialog"
        >
          {paging && <div>
            {paging.error && <p role="alert">Older edits could not be loaded.</p>}
            <button type="button" aria-disabled={paging.loading || !paging.hasOlder}
              onClick={() => { if (!paging.loading && paging.hasOlder) paging.onOlder(); }}>
              {paging.loading ? "Loading older edits…" : paging.hasOlder ? "Load older edits" : "All edits loaded"}
            </button>
          </div>}
          <ol className="lineage-steps">
            {steps.map((step, index) => (
              <li key={`${step.messageId}-${step.artifactId}`}>
                <img
                  src={artifactSource(step.artifactId) ?? undefined}
                  alt={paging?.hasOlder ? "The input to this edit" : `What entered step ${index + 1}`}
                  loading="lazy"
                />
                <div>
                  <strong>{paging?.hasOlder ? "Edit" : `Step ${index + 1}`}</strong>
                  <p>{step.instruction || "No written instruction"}</p>
                </div>
              </li>
            ))}
            <li>
              <img src={resultUrl} alt="The current result" loading="lazy" />
              <div>
                <strong>Result</strong>
                <p>Where the chain stands now.</p>
              </div>
            </li>
          </ol>
        </AccessibleDialog>
      )}
    </>
  );
}
