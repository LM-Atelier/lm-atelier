import { useState } from "react";
import type { Workflow } from "./types";
import "./WorkflowRevisionHistory.css";

function CreationDate({ value }: { value: string }) {
  const date = new Date(value);
  if (!value || !Number.isFinite(date.getTime())) return <small>Creation date unavailable</small>;
  return <time dateTime={value}>{date.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  })}</time>;
}

export function WorkflowRevisionHistory({ workflow, selectedRevisionId, onInspect }: {
  workflow: Workflow;
  selectedRevisionId: string;
  onInspect: (revisionId: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const revisions = expanded ? [...workflow.revisions].sort((a, b) =>
    b.version - a.version || a.id.localeCompare(b.id)) : [];
  return (
    <section className="workflow-revision-history" aria-label="Revision history">
      <h3>Revision history</h3>
      <button className="secondary compact-button" aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}>
        {expanded ? "Hide revision history" : "Show revision history"}
      </button>
      {expanded && (
        <>
          <p>Inspect a saved revision to see its graph, controls, and dependencies. Restoring it is a separate action.</p>
          <ol>{revisions.map((revision) => (
            <li key={revision.id}>
              <button className="secondary compact-button"
                aria-pressed={revision.id === selectedRevisionId}
                onClick={() => onInspect(revision.id)}>
                Inspect v{revision.version}
              </button>
              <CreationDate value={revision.created_at} />
              {revision.id === workflow.current_revision_id && <span className="badge">Current</span>}
              {revision.id === selectedRevisionId && <span className="workflow-history-viewing">Viewing</span>}
            </li>
          ))}</ol>
        </>
      )}
    </section>
  );
}
