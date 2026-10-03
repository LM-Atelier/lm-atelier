import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";

/** Keep one edit as a recipe of the kind Image Studio offers, named and worded before it is saved.
 *
 * The server reads the workflow, model and settings from the run that made the
 * edit, and whether it used a selection, so the recipe is what that edit did
 * rather than what is set up now. It never keeps the selection, a seed, or LoRAs
 * matched to the edit's words. The words start as the ones the person typed for
 * the edit, as Image Studio keeps them, and can be changed before saving.
 */
export function GenerationEditRecipeDialog({ runId, onClose }: { runId: string; onClose: () => void }) {
  const client = useQueryClient();
  const draft = useQuery({
    queryKey: ["edit-recipe-draft", runId],
    queryFn: ({ signal }) => api.editRecipeDraft(runId, signal),
    retry: false,
    staleTime: 0,
    gcTime: 0,
  });
  const [name, setName] = useState("");
  // Untouched, the words are the edit's own; once changed, they are the person's.
  const [edited, setEdited] = useState<string | null>(null);
  const words = edited ?? draft.data?.instruction ?? "";
  const save = useMutation({
    mutationFn: () => api.createEditTemplate({ name: name.trim(), instruction: words.trim(), from_run_id: runId }),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["edit-templates"] }),
  });
  const ready = Boolean(name.trim() && words.trim()) && !save.isPending;
  // Saving removes the form, and the control that was focused with it.
  const done = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (save.data) done.current?.focus();
  }, [save.data]);
  return <AccessibleDialog title="Keep this edit as a recipe" eyebrow="Recipe from an edit"
    closeLabel="Close the recipe" onClose={() => { if (!save.isPending) onClose(); }}
    className="generation-record-dialog">
    {draft.isPending && <p role="status">Reading the edit…</p>}
    {draft.isError && <ErrorCallout message="This edit could not be read, so no recipe can be kept from it." />}
    {draft.data && !save.data && <form className="generation-record-body" onSubmit={(event) => {
      event.preventDefault();
      if (ready) save.mutate();
    }}>
      <p>The recipe keeps the workflow, model and settings this edit ran with, and whether it used a
        selection. The selection itself, the seed and LoRAs matched to the words are not kept. Image Studio
        offers it with its recipes.</p>
      <label>Recipe name
        <input value={name} required maxLength={200} onChange={(event) => setName(event.target.value)} />
      </label>
      <label>Words
        <textarea value={words} required maxLength={20_000} rows={4}
          onChange={(event) => setEdited(event.target.value)} />
      </label>
      {save.error && <p role="alert">{save.error.message}</p>}
      <footer>
        <button type="button" className="secondary" aria-disabled={save.isPending}
          onClick={() => { if (!save.isPending) onClose(); }}>Cancel</button>
        <button type="submit" className="primary" aria-disabled={!ready}>
          {save.isPending ? "Saving…" : "Save recipe"}
        </button>
      </footer>
    </form>}
    {save.data && <>
      <p role="status">Saved the recipe {save.data.name}. Image Studio offers it with its recipes.</p>
      <footer>
        <button ref={done} type="button" className="primary" onClick={onClose}>Done</button>
      </footer>
    </>}
  </AccessibleDialog>;
}
