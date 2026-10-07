import { useEffect, useId, useState } from "react";
import { WorkflowChoiceDropdown } from "./WorkflowChoiceDropdown";
import { useWorkflowRevisionChoice, useWorkflowSummaryPages } from "./useWorkflowReadPages";
import "./GenerationRecordAdaptChoices.css";

export function GenerationRecordAdaptChoices({ operation, value, saving, onChange, onReady }: {
  operation: string; value: string; saving: boolean;
  onChange: (revisionId: string) => void; onReady: (ready: boolean) => void;
}) {
  const id = useId();
  const [search, setSearch] = useState("");
  const pages = useWorkflowSummaryPages(operation, search);
  const role = operation === "text_to_video" || operation === "image_to_video" ? "video" : "image";
  const selected = useWorkflowRevisionChoice(value, role);
  const rows = (pages.data ?? []).filter(row => row.operation === operation && row.current_revision_id);
  const current = selected.data;
  const ready = !value || Boolean(selected.isSuccess && current?.operation === operation
    && current.revision_id === value);
  useEffect(() => { onReady(ready); }, [ready, value, onReady]);
  const options = [
    { value: "", label: "Choose a workflow" },
    ...rows.map(row => ({ value: row.current_revision_id!, label: row.name, searchResult: true })),
  ];
  if (value && !options.some(option => option.value === value)) {
    options.push({ value, label: current?.revision_id === value ? current.workflow_name
      : selected.isPending ? "Selected workflow (loading…)" : "Selected workflow unavailable" });
  }
  return <div className="generation-record-workflow-choice">
    <label htmlFor={id}>Workflow</label>
    <WorkflowChoiceDropdown id={id} label="Workflow" browseLabel="workflows" value={value}
      options={options} browse={{ search, setSearch, pages }} saving={saving}
      onChange={revisionId => {
        if (revisionId === value) return;
        const row = rows.find(item => item.current_revision_id === revisionId);
        if (revisionId && !row && !(ready && revisionId === value)) return;
        onReady(!revisionId);
        onChange(revisionId);
      }} />
    {value && !ready && <p role={selected.isPending ? "status" : "alert"}>
      {selected.isPending ? "Loading selected workflow…" : "The selected workflow is unavailable."}
      {selected.isError && <button type="button" className="secondary compact-button"
        onClick={() => { void selected.refetch(); }}>Retry selected workflow</button>}
    </p>}
  </div>;
}
