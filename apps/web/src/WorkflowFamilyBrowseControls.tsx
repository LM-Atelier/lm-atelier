import { useId } from "react";
import { WorkflowReadPageControls } from "./WorkflowReadPageControls";
import type { WorkflowFamilyBrowseState } from "./useWorkflowFamilyChoices";
import "./WorkflowFamilyBrowseControls.css";

export function WorkflowFamilyBrowseControls({
  browse, label,
}: { browse: WorkflowFamilyBrowseState; label: string }) {
  const searchId = useId();
  return <>
    <label htmlFor={searchId}>Search {label}</label>
    <input id={searchId} className="workflow-family-search" type="search" value={browse.search} maxLength={500}
      onChange={event => browse.setSearch(event.target.value)} />
    <WorkflowReadPageControls pages={browse.pages} label={label} />
  </>;
}
