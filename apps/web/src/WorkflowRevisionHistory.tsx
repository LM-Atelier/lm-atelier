import { useState } from "react";
import type { Workflow } from "./types";
import "./WorkflowRevisionHistory.css";

function CreationDate({ value }: { value: string }) {
  // Stored workflow dates are UTC even when SQLite omits their timezone.
  const timestamp = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?$/.test(value)
    ? `${value}Z` : value;
  const date = new Date(timestamp);
  if (!value || !Number.isFinite(date.getTime())) return <small>Creation date unavailable</small>;
  return <time dateTime={timestamp}>{date.toLocaleString(undefined, {
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
