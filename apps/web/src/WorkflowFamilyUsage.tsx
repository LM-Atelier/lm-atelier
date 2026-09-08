import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import "./WorkflowFamilyUsage.css";

export function WorkflowFamilyUsage({ familyId }: { familyId: string }) {
  const [expanded, setExpanded] = useState(false);
  const usage = useQuery({
    queryKey: ["workflow-family", familyId, "removal-impact"],
    queryFn: () => api.workflowFamilyRemovalImpact(familyId),
    enabled: expanded,
  });
  const report = usage.data;

  return (
    <section className="workflow-family-usage" aria-label="Family usage">
      <h3>Used by</h3>
      <button className="secondary compact-button" aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}>
        {expanded ? "Hide usage" : "Show usage"}
      </button>
      {expanded && (
        <>
          {usage.isPending && <p role="status">Loading usage…</p>}
          {usage.error ? (
            <div>
              <ErrorCallout message={usage.error.message} />
              <button className="secondary compact-button" disabled={usage.isFetching}
                onClick={() => void usage.refetch()}>Retry usage</button>
            </div>
          ) : report && (
            <>
              {usage.isFetching && <p role="status">Refreshing usage…</p>}
              <dl>
                <div><dt>Chat selections</dt><dd>{report.chat_selection_count}</dd></div>
                <div><dt>Project selections</dt><dd>{report.project_selection_count}</dd></div>
                <div><dt>Pinned project revisions</dt><dd>{report.project_revision_pin_count}</dd></div>
                <div><dt>Active runs</dt><dd>{report.active_run_count}</dd></div>
                <div><dt>Queued steps</dt><dd>{report.queued_step_count}</dd></div>
                <div><dt>Historical runs</dt><dd>{report.historical_run_count}</dd></div>
              </dl>
              <p className="muted">Selections and activity are counted separately. Archiving retains accepted work and run history.</p>
            </>
          )}
        </>
      )}
    </section>
  );
}
