import type { WorkflowDestination } from "./useWorkflowDestination";

/** The two destinations, announced as pages rather than tabs.
 *
 * Each one takes you somewhere and the current one is the current page, which
 * is a destination list rather than a tab widget - the same reasoning the
 * Settings navigation follows.
 */
export function WorkflowDestinations({
  current,
  onChoose,
}: {
  current: WorkflowDestination;
  onChoose: (next: WorkflowDestination) => void;
}) {
  return (
    <nav className="workflow-destinations" aria-label="Workflow sections">
      <button className="secondary compact-button" aria-current={current === "library" ? "page" : undefined}
        onClick={() => onChoose("library")}>Library</button>
      <button className="secondary compact-button" aria-current={current === "discover" ? "page" : undefined}
        onClick={() => onChoose("discover")}>Discover</button>
    </nav>
  );
}
