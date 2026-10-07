import type { Ref } from "react";
import { ProfilePicker } from "./ProfilePicker";
import { ReadyWorkflowBrowseControls } from "./ReadyWorkflowBrowseControls";
import type { ComparisonChoiceDraft } from "./generationComparison";
import type { ExperimentOperation } from "./generationExperimentTypes";
import { useProfileLibrary } from "./useProfileLibrary";
import { useReadyWorkflowChoices } from "./useReadyWorkflowChoices";
import "./GenerationComparisonView.css";

/** One choice: its label, the model it uses and the exact workflow revision it runs. */
export function ComparisonChoiceFields({ legend, value, onChange, labelRef, operation = "text_to_image" }: {
  legend: string;
  /** What the workflow must do: make a picture or a video from words, or change the picture given. */
  operation?: ExperimentOperation;
  value: ComparisonChoiceDraft;
  onChange: (next: ComparisonChoiceDraft) => void;
  labelRef?: Ref<HTMLInputElement>;
}) {
  const video = operation === "text_to_video";
  const library = useProfileLibrary(video ? "video" : "image", value.profileId || null);
  const workflows = useReadyWorkflowChoices(value.revisionId ? [value.revisionId] : [], operation);
  const choices = workflows.rows.map((row) => ({
    id: row.revision_id, label: `${row.family_name} - ${row.workflow_name} · v${row.revision_version}`,
  }));
  return <fieldset className="comparison-choice">
    <legend>{legend}</legend>
    <label>Label<input ref={labelRef} value={value.label} maxLength={80}
      onChange={(event) => onChange({ ...value, label: event.target.value })} /></label>
    <ProfilePicker library={library} label="Model" searchLabel="Search models" value={value.profileId}
      selectedId={value.profileId || null} onChange={(profileId) => onChange({ ...value, profileId })}>
      <option value="">{video ? "Choose a video model" : "Choose an image model"}</option>
    </ProfilePicker>
    <ReadyWorkflowBrowseControls workflows={workflows} />
    <label>Workflow<select aria-label="Workflow" value={value.revisionId}
      onChange={(event) => onChange({ ...value, revisionId: event.target.value })}>
      <option value="">{operation === "image_to_image" ? "Choose a ready image editing workflow"
        : video ? "Choose a ready video workflow" : "Choose a ready image workflow"}</option>
      {value.revisionId && !choices.some((choice) => choice.id === value.revisionId)
        && <option value={value.revisionId}>{workflows.missingLabel}</option>}
      {choices.map((choice) => <option key={choice.id} value={choice.id}>{choice.label}</option>)}
    </select></label>
  </fieldset>;
}
