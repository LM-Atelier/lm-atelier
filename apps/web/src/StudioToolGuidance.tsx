/** What a tool that cannot run says, and where it sends you.
 *
 * Naming what is missing is only half an answer. The other half is the place
 * that fixes it, which was previously left as an exercise: the sentence said
 * "install an inpainting workflow" and then stopped, as though finding one
 * were the easy part.
 */
export function StudioToolGuidance({
  reason,
  onOpenWorkflows,
}: {
  reason: string;
  onOpenWorkflows: () => void;
}) {
  return (
    <p className="studio-tool-guidance" id="studio-tool-guidance" role="status">
      {reason}{" "}
      <button className="link-button" onClick={onOpenWorkflows}>
        Browse workflows
      </button>
    </p>
  );
}

/** What a model's tool says before the Studio knows whether its workflow is here.
 *
 * Apply waits for the answer rather than guessing. Guessing yes made a tool
 * whose workflow was missing look ready, and the gap was found only when the
 * edit was refused, after the selection had been drawn.
 */
export function StudioCapabilityCheck({ failed, onRetry }: { failed: boolean; onRetry: () => void }) {
  return (
    <p className="studio-tool-guidance" role="status">
      {failed ? "Could not check whether this tool can run here." : "Checking whether this tool can run here…"}
      {failed && (
        <>
          {" "}
          <button className="link-button" onClick={onRetry}>
            Try again
          </button>
        </>
      )}
    </p>
  );
}
