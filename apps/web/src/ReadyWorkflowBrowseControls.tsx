import { useId } from "react";
import { WorkflowReadPageControls } from "./WorkflowReadPageControls";
import type { useReadyWorkflowChoices } from "./useReadyWorkflowChoices";

export function ReadyWorkflowBrowseControls({
  workflows,
}: { workflows: ReturnType<typeof useReadyWorkflowChoices> }) {
  const searchId = useId();
  return <>
    <label htmlFor={searchId}>Search ready image workflows</label>
    <input id={searchId} type="search" maxLength={500} value={workflows.search}
      onChange={event => workflows.setSearch(event.target.value)} />
    <WorkflowReadPageControls pages={workflows.pages} label="ready image workflows" />
    {workflows.hasSelection && workflows.selected.isPending
      && <p role="status">Checking selected workflows…</p>}
    {workflows.hasSelection && workflows.selected.isError && <p role="alert">
      Selected workflows could not be checked. Pinned workflows are kept as they are.{" "}
      <button type="button" className="secondary compact-button"
        onClick={() => void workflows.selected.refetch()}>Retry selected workflows</button>
    </p>}
  </>;
}
