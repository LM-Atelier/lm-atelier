import { useMutation, type UseQueryResult } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { recipeDraftPayload } from "./generationComparison";
import type { RecipeDraft } from "./recipeDraftTypes";
import type { WorkflowSelectorCapability } from "./types";
import { WorkflowRecipeEditor } from "./WorkflowRecipeEditor";
import type { WorkflowUseCasePreset, WorkflowUseCasePresetCreate } from "./workflowUseCaseTypes";
import "./GenerationComparisonView.css";
import "./WorkflowRecipeManager.css";

/** Review a drafted recipe in the ordinary recipe editor, then save it.
 *
 * A recipe holds settings only, so the model and workflow are named beside it,
 * and a new chat can be set up with all three. Nothing here changes a default
 * or what Automatic chooses.
 */
export function RecipeDraftDialog({ title, eyebrow, draft, failure, onOpenChat, onClose }: {
  title: string;
  eyebrow: string;
  draft: UseQueryResult<RecipeDraft>;
  /** What the person is told when no draft could be made. */
  failure: (error: Error) => string;
  /** Show a chat; without it the recipe is saved and no chat is offered. */
  onOpenChat?: (chatId: string) => void;
  onClose: () => void;
}) {
  const save = useMutation({
    mutationFn: (payload: WorkflowUseCasePresetCreate) => api.createWorkflowUseCasePreset(payload),
  });
  const newChat = useMutation({
    mutationFn: ({ recipe, from }: { recipe: WorkflowUseCasePreset; from: RecipeDraft }) =>
      chatWithSetup(recipe, from),
    onSuccess: (chatId) => {
      onClose();
      onOpenChat?.(chatId);
    },
  });
  const busy = save.isPending || newChat.isPending;
  const made = draft.data;
  const saved = save.data;
  // The model and workflow fit the drafted use case; a recipe saved for
  // another, or turned off, could not be chosen in the chat they set up.
  const offerChat = Boolean(onOpenChat && made && saved?.enabled && saved.use_case === made.use_case);
  // Saving removes the editor, and the control that was focused with it.
  const savedAction = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (saved) savedAction.current?.focus();
  }, [saved]);
  return <AccessibleDialog title={title} eyebrow={eyebrow}
    closeLabel="Close the recipe" onClose={() => { if (!busy) onClose(); }}
    // The recipe manager's own look, since its editor is what is inside.
    className="workflow-recipe-manager comparison-recipe">
    {draft.isPending && <p role="status">Drafting the recipe…</p>}
    {draft.isError && <ErrorCallout message={failure(draft.error)} />}
    {made && !saved && <>
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
    {made && saved && <>
      <p role="status">Saved the recipe {saved.name}.</p>
      {offerChat && <button ref={savedAction} type="button" className="primary" aria-disabled={busy}
        onClick={() => { if (!busy) newChat.mutate({ recipe: saved, from: made }); }}>
        {newChat.isPending ? "Setting up the chat…" : "Use in a new chat"}</button>}
      <ErrorCallout message={newChat.error ? newChat.error.message : null} />
      <button ref={offerChat ? undefined : savedAction} type="button" className="secondary" aria-disabled={busy}
        onClick={() => { if (!busy) onClose(); }}>Done</button>
    </>}
  </AccessibleDialog>;
}

/** A new chat set to the draft's model, its workflow's family and the saved recipe; its id once all three are set. */
async function chatWithSetup(recipe: WorkflowUseCasePreset, from: RecipeDraft): Promise<string> {
  const capability = recipeCapability(from.use_case);
  const chat = await api.createChat(null);
  try {
    if (from.profile_id) {
      await api.updateChat(chat.id, capability === "video"
        ? { active_video_profile_id: from.profile_id }
        : { active_image_profile_id: from.profile_id });
    }
    if (from.workflow_family_id) {
      await api.setChatWorkflowSelection(chat.id, capability, { mode: "family", workflow_family_id: from.workflow_family_id });
    }
    await api.setWorkflowUseCaseChoice({ kind: "chat", id: chat.id }, from.use_case, { mode: "preset", preset_id: recipe.id });
  } catch (error) {
    const reason = error instanceof Error ? ` ${error.message}` : "";
    throw new Error(`A new chat was made, but not all of this setup could be applied to it.${reason}`, { cause: error });
  }
  return chat.id;
}

/** Which of a chat's models and workflows a recipe's use case is chosen beside. */
function recipeCapability(useCase: RecipeDraft["use_case"]): Extract<WorkflowSelectorCapability, "image" | "video"> {
  return useCase === "image_generation" ? "image" : "video";
}
