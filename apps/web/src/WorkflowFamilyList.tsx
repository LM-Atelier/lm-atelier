import { useState } from "react";
import "./WorkflowFamilyList.css";
import type { Workflow, WorkflowFamily, WorkflowVariantReadiness } from "./types";

const readinessLabels: Record<WorkflowVariantReadiness, string> = {
  ready: "Ready",
  setup_required: "Needs setup",
  review_required: "Needs review",
  unavailable: "Unavailable",
};
const operationLabels: Record<string, string> = {
  text_to_image: "Text to image",
  image_to_image: "Image to image",
  text_to_video: "Text to video",
  image_to_video: "Image to video",
  video_to_video: "Video to video",
};

interface Props {
  families: WorkflowFamily[];
  workflows: Workflow[];
  selectedId: string | null;
  onSelect: (workflow: Workflow) => void;
  includeArchived: boolean;
  onIncludeArchivedChange: (include: boolean) => void;
  loading: boolean;
}

export function WorkflowFamilyList({
  families, workflows, selectedId, onSelect, includeArchived, onIncludeArchivedChange, loading,
}: Props) {
  const [search, setSearch] = useState("");
  const [operation, setOperation] = useState("");
  const [readiness, setReadiness] = useState("");
  const [defaultsOnly, setDefaultsOnly] = useState(false);
  const query = search.trim().toLocaleLowerCase();
  const matchesSearch = (values: string[]) => values.some((value) => value.toLocaleLowerCase().includes(query));
  const definitions = new Map(workflows.map((workflow) => [workflow.id, workflow]));
  const represented = new Set(families.flatMap((family) => family.variants
    .filter((variant) => {
      const workflow = definitions.get(variant.id);
      return workflow && (workflow.family_id === undefined || workflow.family_id === family.id);
    }).map((variant) => variant.id)));
  const operations = [...new Set([
    ...families.flatMap((family) => family.variants.map((variant) => variant.operation)),
    ...workflows.map((workflow) => workflow.operation),
  ])].sort();
  const visibleFamilies = families.filter((family) => includeArchived || !family.archived)
    .filter((family) => !defaultsOnly || family.preferences.some((preference) => preference.is_default))
    .filter((family) => matchesSearch([
      family.name, family.description, family.use_case, ...family.tags,
      ...family.variants.map((variant) => variant.name),
    ]))
    .map((family) => ({ family, variants: family.variants.filter((variant) =>
      (!operation || variant.operation === operation) && (!readiness || variant.readiness === readiness)) }))
    .filter(({ variants }) => variants.length > 0)
    .sort((a, b) => a.family.name.localeCompare(b.family.name) || a.family.id.localeCompare(b.family.id));
  const remaining = workflows.filter((workflow) => !represented.has(workflow.id)
    && workflow.family_id == null
    && !defaultsOnly && !readiness
    && (!operation || workflow.operation === operation)
    && matchesSearch([workflow.name, workflow.description]))
    .sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
  const ungrouped = remaining.filter((workflow) => workflow.family_id === null);
  const other = remaining.filter((workflow) => workflow.family_id === undefined);
  const definitionButton = (workflow: Workflow) => (
    <button key={workflow.id} className={selectedId === workflow.id ? "selected" : ""}
      aria-pressed={selectedId === workflow.id} onClick={() => onSelect(workflow)}>
      <span><strong>{workflow.name}</strong><small>{operationLabels[workflow.operation] ?? workflow.operation}
        {" · "}{workflow.revisions.length} revision{workflow.revisions.length === 1 ? "" : "s"}</small></span>
    </button>
  );

  return (
    <div className="workflow-family-browser">
      <div className="workflow-family-filters">
        <label>Search workflow families
          <input type="search" value={search} onChange={(event) => setSearch(event.target.value)} />
        </label>
        <label>Filter by operation
          <select value={operation} onChange={(event) => setOperation(event.target.value)}>
            <option value="">All operations</option>
            {operations.map((value) => <option key={value} value={value}>{operationLabels[value] ?? value}</option>)}
          </select>
        </label>
        <label>Filter by readiness
          <select value={readiness} onChange={(event) => setReadiness(event.target.value)}>
            <option value="">All readiness states</option>
            {Object.entries(readinessLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label className="workflow-family-checkbox"><input type="checkbox" checked={defaultsOnly} onChange={(event) => setDefaultsOnly(event.target.checked)} /> Defaults only</label>
        <label className="workflow-family-checkbox"><input type="checkbox" checked={includeArchived} onChange={(event) => onIncludeArchivedChange(event.target.checked)} /> Show archived families</label>
      </div>
      {loading ? <p role="status">Loading workflow families…</p> : (
        <>
          {visibleFamilies.map(({ family, variants }) => (
            <section key={family.id} aria-labelledby={`workflow-family-${family.id}`}>
              <h3 id={`workflow-family-${family.id}`}>{family.name}</h3>
              {(family.use_case || family.description) && <p className="muted">{family.use_case || family.description}</p>}
              {family.archived && <span className="badge">Archived</span>}
              <div className="workflow-list">
                {variants.map((variant) => {
                  const workflow = definitions.get(variant.id);
                  const selectable = workflow && (workflow.family_id === undefined || workflow.family_id === family.id);
                  return (
                    <button key={variant.id} disabled={!selectable}
                      className={selectedId === variant.id && selectable ? "selected" : ""}
                      aria-pressed={selectedId === variant.id && Boolean(selectable)}
                      onClick={() => { if (selectable) onSelect(workflow); }}>
                      <span><strong>{variant.name}</strong>
                        <small>{operationLabels[variant.operation] ?? variant.operation}
                          {variant.current_revision_version !== null && ` · v${variant.current_revision_version}`}</small>
                        <span className="badge">{readinessLabels[variant.readiness]}</span>
                      </span>
                    </button>
                  );
                })}
              </div>
            </section>
          ))}
          {ungrouped.length > 0 && <section aria-label="Ungrouped workflows">
            <h3>Ungrouped workflows</h3><div className="workflow-list">{ungrouped.map(definitionButton)}</div>
          </section>}
          {other.length > 0 && <section aria-label="Other workflow definitions">
            <h3>Other workflow definitions</h3><div className="workflow-list">{other.map(definitionButton)}</div>
          </section>}
          {visibleFamilies.length === 0 && remaining.length === 0 && <p>No workflows match these filters.</p>}
        </>
      )}
    </div>
  );
}
