import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import type { WorkflowDependencyResourceKind } from "./types";
import "./WorkflowFamilyDependencies.css";

const resourceLabels: Record<WorkflowDependencyResourceKind, string> = {
  model_profile: "Model profiles",
  model_install: "Model files",
  model_asset: "Model assets",
  custom_node: "Custom nodes",
  registry_package: "Registry packages",
  runtime: "Runtimes",
};
const resourceKinds = Object.keys(resourceLabels) as WorkflowDependencyResourceKind[];

export function WorkflowFamilyDependencies({ familyId }: { familyId: string }) {
  const [expanded, setExpanded] = useState(false);
  const [currentOnly, setCurrentOnly] = useState(false);
  const dependencies = useQuery({
    queryKey: ["workflow-family", familyId, "removal-impact"],
    queryFn: () => api.workflowFamilyRemovalImpact(familyId),
    enabled: expanded,
  });
  const report = dependencies.data;
  const visible = report?.dependencies.filter((item) => !currentOnly || item.current_revision) ?? [];

  return (
    <section className="workflow-family-dependencies" aria-label="Family dependencies">
      <h3>Dependencies</h3>
      <button className="secondary compact-button" aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}>
        {expanded ? "Hide dependencies" : "Show dependencies"}
      </button>
      {expanded && (
        <>
          {dependencies.isPending && <p role="status">Loading dependencies…</p>}
          {dependencies.error ? (
            <div>
              <ErrorCallout message={dependencies.error.message} />
              <button className="secondary compact-button" disabled={dependencies.isFetching}
                onClick={() => void dependencies.refetch()}>Retry dependencies</button>
            </div>
          ) : report && (
            <>
              {dependencies.isFetching && <p role="status">Refreshing dependencies…</p>}
              <p className="muted">Recorded bindings across this family's revisions. Workflow readiness is shown on each variant.</p>
              <label className="workflow-family-dependency-filter">
                <input type="checkbox" checked={currentOnly} onChange={(event) => setCurrentOnly(event.target.checked)} />
                Current revisions only
              </label>
              {visible.length === 0 && <p>{currentOnly ? "No bindings in current revisions." : "No recorded dependency bindings."}</p>}
              {resourceKinds.map((kind) => {
                const resources = visible.filter((item) => item.resource_kind === kind);
                return resources.length > 0 && (
                  <section key={kind}>
                    <h4>{resourceLabels[kind]}</h4>
                    <ul>
                      {resources.map((resource) => (
                        <li key={resource.resource_id}>
                          <strong>{resource.resource_name || resource.resource_id}</strong>
                          <span className="badge">{resource.current_revision ? "Current revision" : "Earlier revisions only"}</span>
                          <small>{resource.binding_count} {resource.binding_count === 1 ? "binding" : "bindings"} across {resource.revision_count} {resource.revision_count === 1 ? "revision" : "revisions"}</small>
                          {resource.shared && <small>Also used by {resource.other_workflow_count} other {resource.other_workflow_count === 1 ? "workflow" : "workflows"}</small>}
                        </li>
                      ))}
                    </ul>
                  </section>
                );
              })}
            </>
          )}
        </>
      )}
    </section>
  );
}
