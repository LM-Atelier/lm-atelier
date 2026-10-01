import { useState } from "react";
import { WorkflowFamilyCard } from "./WorkflowFamilyCard";
import { workflowOperationLabels } from "./workflowFamilies";
import type { WorkflowLibraryChoice } from "./WorkflowFamilyCard";
import { WorkflowReadPageControls } from "./WorkflowReadPageControls";
import { EMPTY_LIBRARY_FILTERS, useWorkflowLibraryReads } from "./useWorkflowLibraryReads";
import type { WorkflowLibraryFilters } from "./useWorkflowLibraryReads";
import type { WorkflowInstallOffer, WorkflowVariantReadiness } from "./types";
import "./WorkflowFamilyList.css";

const readinessLabels: Record<WorkflowVariantReadiness, string> = {
  ready: "Ready", setup_required: "Needs setup", review_required: "Needs review", unavailable: "Unavailable",
};

interface Props {
  selectedId: string | null;
  onSelect: (workflow: WorkflowLibraryChoice) => void;
  includeArchived: boolean;
  onIncludeArchivedChange: (include: boolean) => void;
  onReviewInstall?: (offer: WorkflowInstallOffer, workflowName: string) => void;
  /** The workflow whose installation the details panel shows in full. */
  installationInDetails?: string | null;
}

export function WorkflowFamilyList({ selectedId, onSelect, includeArchived, onIncludeArchivedChange,
  onReviewInstall, installationInDetails }: Props) {
  const [filters, setFilters] = useState<WorkflowLibraryFilters>(EMPTY_LIBRARY_FILTERS);
  const { families, ungrouped, operations, showUngrouped } = useWorkflowLibraryReads(filters, includeArchived);
  const loading = families.isPending || (showUngrouped && ungrouped.isLoading);
  const rows = families.data ?? [];
  const other = showUngrouped ? ungrouped.data ?? [] : [];
  const update = (value: Partial<WorkflowLibraryFilters>) => setFilters(previous => ({ ...previous, ...value }));
  return <div className="workflow-family-browser">
    <div className="workflow-family-filters">
      <label>Search workflow families
        <input type="search" value={filters.search} maxLength={500}
          onChange={event => update({ search: event.target.value })} />
      </label>
      <label>Filter by operation
        <select value={filters.operation} onChange={event => update({ operation: event.target.value })}>
          <option value="">All operations</option>
          {(operations.data ?? []).map(value => <option key={value} value={value}>{workflowOperationLabels[value] ?? value}</option>)}
        </select>
      </label>
      <label>Filter by readiness
        <select value={filters.readiness}
          onChange={event => update({ readiness: event.target.value as WorkflowLibraryFilters["readiness"] })}>
          <option value="">All readiness states</option>
          {Object.entries(readinessLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </label>
      <label>Family source
        <select value={filters.source} onChange={event => update({ source: event.target.value as WorkflowLibraryFilters["source"] })}>
          <option value="">All sources</option><option value="profile">From model profiles</option>
          <option value="workflow">Other workflow families</option>
        </select>
      </label>
      <label>Sort workflow families
        <select value={filters.order} onChange={event => update({ order: event.target.value as WorkflowLibraryFilters["order"] })}>
          <option value="name">Name</option><option value="readiness">Readiness</option>
        </select>
      </label>
      <label className="workflow-family-checkbox"><input type="checkbox" checked={filters.defaultsOnly}
        onChange={event => update({ defaultsOnly: event.target.checked })} /> Defaults only</label>
      <label className="workflow-family-checkbox"><input type="checkbox" checked={includeArchived}
        onChange={event => onIncludeArchivedChange(event.target.checked)} /> Show archived families</label>
    </div>
    {operations.error && <div role="alert">{operations.error.message}{" "}
      <button type="button" className="secondary compact-button"
        onClick={() => void operations.refetch()}>Retry operation choices</button>
    </div>}
    {rows.map(family => <WorkflowFamilyCard key={family.id} family={family} selectedId={selectedId}
      onSelect={onSelect} operation={filters.operation || undefined} readiness={filters.readiness || undefined}
      onReviewInstall={onReviewInstall} installationInDetails={installationInDetails} />)}
    <WorkflowReadPageControls pages={families} label="workflow families" />
    {showUngrouped && <>
      {other.length > 0 && <section aria-label="Ungrouped workflows">
        <h3>Ungrouped workflows</h3><div className="workflow-list">{other.map(workflow => (
          <button key={workflow.id} className={selectedId === workflow.id ? "selected" : ""}
            aria-pressed={selectedId === workflow.id} onClick={() => onSelect(workflow)}>
            <span><strong>{workflow.name}</strong><small>{workflowOperationLabels[workflow.operation] ?? workflow.operation}
              {" · "}{workflow.revision_count} revision{workflow.revision_count === 1 ? "" : "s"}</small></span>
          </button>
        ))}</div>
      </section>}
      <WorkflowReadPageControls pages={ungrouped} label="ungrouped workflows" />
    </>}
    {!loading && !families.error && !(showUngrouped && ungrouped.error)
      && rows.length === 0 && other.length === 0 && <p>No workflows match these filters.</p>}
  </div>;
}
