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
