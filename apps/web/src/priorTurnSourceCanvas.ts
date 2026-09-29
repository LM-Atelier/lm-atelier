import { priorTurnEditConfigurationSource, selectPriorTurnEditConfiguration, type PriorTurnEditDraft } from "./priorTurnEditDraft";
import { workflowRevisionForTurn } from "./turnEditorContext";
import type { WorkflowFamily, WorkflowSelection } from "./types";

/** Resolve the image configuration independently of the settings page being viewed. */
export function priorTurnSourceCanvasRevision(draft: PriorTurnEditDraft, families: WorkflowFamily[]): string | null {
  // Source canvases currently apply to one image step; ordered plans need their own binding.
  if ((draft.source.steps?.length ?? 0) > 1) return null;
  const image = draft.settingsRole === "image" ? draft : selectPriorTurnEditConfiguration(draft, { role: "image" });
  const inherited = priorTurnEditConfigurationSource(image);
  const roleChoice = image.configurations && "stepId" in image.configurations.target
    ? image.configurations.roles.image?.workflow_selection : undefined;
  const choice = image.workflowChoice.kind === "explicit" ? image.workflowChoice.value : roleChoice;
  if (choice === undefined) return inherited?.workflow_revision_id ?? null;
  // "Current selection" cannot borrow the original revision while its current scope is unknown.
  if (choice === null) return null;
  const selection: WorkflowSelection = {
    selector_capability: choice.selector_capability, mode: choice.mode,
    workflow_family_id: choice.mode === "family" ? choice.workflow_family_id : null,
    workflow_revision_id: choice.mode === "revision" ? choice.workflow_revision_id : null,
    legacy_profile_id: null,
  };
  return workflowRevisionForTurn("image", true, families, selection, null);
}
