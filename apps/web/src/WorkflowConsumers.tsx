import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { WorkflowDependencyResourceKind } from "./types";

/** Which workflows would stop working without this file.
 *
 * Deleting a model has always said what it removes and never what it
 * breaks, so the answer arrived later as a workflow that used to run and
 * now does not. This asks the server that question at the point the
 * decision is made.
 *
 * Silence is deliberate when nothing uses it: an empty list under every
 * delete would train people to skip reading the ones that matter.
 */
export function WorkflowConsumers({
  kind,
  resourceId,
}: {
  kind: WorkflowDependencyResourceKind;
  resourceId: string;
}) {
  const consumers = useQuery({
    queryKey: ["workflow-consumers", kind, resourceId],
    queryFn: () => api.workflowResourceConsumers(kind, resourceId),
  });

  const found = consumers.data?.consumers ?? [];
  if (consumers.isLoading || found.length === 0) return null;

  // A workflow whose *current* revision needs this breaks the next time it
  // runs. One where only an older revision needs it keeps working, and
  // saying so is the difference between a warning and an alarm.
  const current = found.filter((consumer) => consumer.current_revision);
  const older = found.length - current.length;

  return (
    <div className="workflow-consumers">
      <h4>
        {current.length > 0
          ? `${current.length} workflow${current.length === 1 ? "" : "s"} needs this to run`
          : "Only older revisions use this"}
      </h4>
      <ul>
        {(current.length > 0 ? current : found).map((consumer) => (
          <li key={consumer.workflow_id}>
            {consumer.workflow_name}
            {consumer.workflow_family_name ? <small> {consumer.workflow_family_name}</small> : null}
          </li>
        ))}
      </ul>
      {current.length > 0 && older > 0 && (
        <small>
          {older} older revision{older === 1 ? "" : "s"} also references it and would stay as it is.
        </small>
      )}
    </div>
  );
}
