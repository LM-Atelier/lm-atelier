import { useState } from "react";
import type { WorkflowRevision } from "./types";
import "./WorkflowRevisionComparison.css";

type Change = { section: string; path: string; before: unknown; after: unknown };
type Pending = Change;
const MAX_CHANGES = 100;
const MAX_FIELDS = 10000;

function container(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}

function compare(selected: WorkflowRevision, current: WorkflowRevision) {
  const sections: [string, unknown, unknown][] = [
    ["Executable graph", selected.api_graph_json, current.api_graph_json],
    ["Declared controls", selected.input_schema_json, current.input_schema_json],
    ["Dependencies", selected.dependencies_json, current.dependencies_json],
    ["Editor layout", selected.ui_graph_json, current.ui_graph_json],
    ["Runtime", { engine: selected.engine, engine_version: selected.engine_version },
      { engine: current.engine, engine_version: current.engine_version }],
  ];
  const pending: Pending[] = sections.reverse().map(([section, before, after]) =>
    ({ section, path: "", before, after }));
  const changes: Change[] = [];
  let fields = 0;
  while (pending.length && changes.length < MAX_CHANGES && fields < MAX_FIELDS) {
    const item = pending.pop()!;
    fields += 1;
    if (Object.is(item.before, item.after)) continue;
    if (container(item.before) && container(item.after)
      && Array.isArray(item.before) === Array.isArray(item.after)) {
      const keys = [...new Set([...Object.keys(item.before), ...Object.keys(item.after)])].sort();
      for (const key of keys.reverse()) {
        const segment = key.replaceAll("~", "~0").replaceAll("/", "~1");
        pending.push({
          section: item.section, path: item.path + "/" + segment,
          before: Object.hasOwn(item.before, key) ? item.before[key] : undefined,
          after: Object.hasOwn(item.after, key) ? item.after[key] : undefined,
        });
      }
    } else {
      changes.push(item);
    }
  }
  return { changes, limited: pending.length > 0 };
}

function Value({ value }: { value: unknown }) {
  if (value === undefined) return <span className="workflow-change-missing">Not present</span>;
  return <pre>{JSON.stringify(value, null, 2)}</pre>;
}

export function WorkflowRevisionComparison({ selected, current }: {
  selected: WorkflowRevision;
  current: WorkflowRevision;
}) {
  const [expanded, setExpanded] = useState(false);
  const result = expanded ? compare(selected, current) : null;
  return (
    <section className="workflow-revision-comparison">
      <h3>Compare with current revision</h3>
      <button className="secondary compact-button" aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}>
        {expanded ? "Hide revision changes" : `Show changes from v${selected.version} to v${current.version}`}
      </button>
      {result && (
        <section aria-label="Revision changes">
          <p>Changes from selected v{selected.version} to current v{current.version}.
            Field paths identify saved values; array positions are compared in order.</p>
          {result.limited && <p role="status">Comparison limit reached. Inspect the saved graphs,
            controls, and dependencies for the remaining content.</p>}
          {!result.changes.length && !result.limited
            ? <p>No content differences in these revisions.</p>
            : result.changes.length > 0 && (
              <div className="workflow-revision-change-table">
                <table>
                  <thead><tr><th scope="col">Content and field</th>
                    <th scope="col">Selected v{selected.version}</th>
                    <th scope="col">Current v{current.version}</th></tr></thead>
                  <tbody>{result.changes.map((change) => (
                    <tr key={change.section + ":" + change.path}>
                      <th scope="row"><span>{change.section}</span><code>{change.path || "/"}</code></th>
                      <td><Value value={change.before} /></td><td><Value value={change.after} /></td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            )}
        </section>
      )}
    </section>
  );
}
