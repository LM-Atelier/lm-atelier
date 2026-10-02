import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { comparisonFailure, recipeDraftPayload } from "./generationComparison";
import type { ExperimentArm, GenerationExperimentRecipeDraft } from "./generationExperimentTypes";
import { WorkflowRecipeEditor } from "./WorkflowRecipeEditor";
import type { WorkflowUseCasePreset, WorkflowUseCasePresetCreate } from "./workflowUseCaseTypes";
import "./GenerationComparisonView.css";
import "./WorkflowRecipeManager.css";

/** Keep the setup one choice ran with: its settings as a recipe, reviewed before it is saved.
 *
 * A recipe holds settings only, so the model and workflow are named beside it,
 * and a new chat can be set up with all three. Nothing here changes a default
 * or what Automatic chooses.
 */
export function ComparisonKeepRecipe({ experimentId, arm, onOpenChat }: {
  experimentId: string;
  arm: ExperimentArm;
  /** Show a chat; without it the recipe is saved and no chat is offered. */
  onOpenChat?: (chatId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  return <>
    <button type="button" className="secondary compact-button" onClick={() => setOpen(true)}>Keep as a recipe</button>
    {open && <ComparisonRecipeDialog experimentId={experimentId} arm={arm} onOpenChat={onOpenChat}
      onClose={() => setOpen(false)} />}
  </>;
}

function ComparisonRecipeDialog({ experimentId, arm, onOpenChat, onClose }: {
  experimentId: string;
  arm: ExperimentArm;
  onOpenChat?: (chatId: string) => void;
  onClose: () => void;
}) {
  const draft = useQuery({
    queryKey: ["generation-experiments", experimentId, "recipe-draft", arm.ordinal],
    queryFn: ({ signal }) => api.generationExperimentRecipeDraft(experimentId, arm.ordinal, signal),
  });
  const save = useMutation({
    mutationFn: (payload: WorkflowUseCasePresetCreate) => api.createWorkflowUseCasePreset(payload),
  });
  const newChat = useMutation({
    mutationFn: ({ recipe, from }: { recipe: WorkflowUseCasePreset; from: GenerationExperimentRecipeDraft }) =>
      chatWithSetup(recipe, from),
    onSuccess: (chatId) => {
      onClose();
      onOpenChat?.(chatId);
    },
  });
  const busy = save.isPending || newChat.isPending;
  const made = draft.data;
  return <AccessibleDialog title={`Keep ${arm.label} as a recipe`} eyebrow="Recipe from a comparison"
    closeLabel="Close the recipe" onClose={() => { if (!busy) onClose(); }}
    // The recipe manager's own look, since its editor is what is inside.
    className="workflow-recipe-manager comparison-recipe">
    {draft.isPending && <p role="status">Drafting the recipe…</p>}
    {draft.isError && <ErrorCallout message={comparisonFailure(draft.error).message} />}
    {made && !save.data && <>
      <p>Made with {made.profile_name ?? "its model"} on {made.workflow_name}
        {made.workflow_version === null ? "" : ` (version ${made.workflow_version})`}. A recipe keeps the
        settings only; choose this model and workflow where you use it.</p>
      {made.left_out.length > 0 && <div className="comparison-recipe-left-out">
        <p>Not in the recipe:</p>
        <ul>{made.left_out.map((item) => <li key={item.setting}><strong>{item.setting}</strong>: {item.message}</li>)}</ul>
      </div>}
      <WorkflowRecipeEditor recipe={null} draft={recipeDraftPayload(made)} referenceWorkflowId={made.workflow_id}
        saving={save.isPending} error={save.error}
        onSave={(payload) => { if (!busy) save.mutate(payload); }}
        onCancel={() => { if (!busy) onClose(); }} />
    </>}
    {made && save.data && <>
      <p role="status">Saved the recipe {save.data.name}.</p>
      {onOpenChat && <button type="button" className="primary" aria-disabled={busy}
        onClick={() => { if (!busy && save.data) newChat.mutate({ recipe: save.data, from: made }); }}>
        {newChat.isPending ? "Setting up the chat…" : "Use in a new chat"}</button>}
      <ErrorCallout message={newChat.error ? newChat.error.message : null} />
      <button type="button" className="secondary" aria-disabled={busy} onClick={() => { if (!busy) onClose(); }}>Done</button>
    </>}
  </AccessibleDialog>;
}

/** A new chat set to the choice's model, its workflow's family and the saved recipe; its id once all three are set. */
async function chatWithSetup(recipe: WorkflowUseCasePreset, from: GenerationExperimentRecipeDraft): Promise<string> {
  const chat = await api.createChat(null);
  try {
    await api.updateChat(chat.id, { active_image_profile_id: from.profile_id });
    if (from.workflow_family_id) {
      await api.setChatWorkflowSelection(chat.id, "image", { mode: "family", workflow_family_id: from.workflow_family_id });
    }
    await api.setWorkflowUseCaseChoice({ kind: "chat", id: chat.id }, "image_generation", { mode: "preset", preset_id: recipe.id });
  } catch (error) {
    const reason = error instanceof Error ? ` ${error.message}` : "";
    throw new Error(`A new chat was made, but not all of this setup could be applied to it.${reason}`, { cause: error });
  }
  return chat.id;
}
