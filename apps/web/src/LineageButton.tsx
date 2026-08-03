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
}: {
  steps: EditLineageStep[];
  resultUrl: string;
}) {
  const [open, setOpen] = useState(false);
  if (steps.length < 2) return null;
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
          eyebrow={`${steps.length} steps`}
          closeLabel="Close lineage"
          onClose={() => setOpen(false)}
          className="lineage-dialog"
        >
          <ol className="lineage-steps">
            {steps.map((step, index) => (
              <li key={`${step.messageId}-${step.artifactId}`}>
                <img
                  src={artifactSource(step.artifactId) ?? undefined}
                  alt={`What entered step ${index + 1}`}
                  loading="lazy"
                />
                <div>
                  <strong>Step {index + 1}</strong>
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
