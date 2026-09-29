import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { StudioRecipeSource } from "./studioRecipeSource";
import type { EditTemplate } from "./types";

/** Saved recipes, offered where the edit is being made, and a way to save one from a result.
 *
 * A recipe records the workflow, the model, the settings, and whether the edit
 * expects a selection. Until now the only way to reach one was the chat
 * composer's template dialog, which predates this view entirely - so the place
 * that saves recipes could not use them.
 *
 * A recipe whose workflow is not installed is still shown, and says so. The
 * alternative is hiding it, which reads as "you never saved that" rather than
 * "the thing it needs is missing".
 *
 * A result a model made can be saved as a recipe here. The server reads it from
 * the run that made the result, so it records what that run did, not what is set
 * up when Save is pressed, and it never keeps the selection or the seed.
 */
export function StudioRecipes({
  onApply,
  disabled = false,
  from = null,
}: {
  onApply: (recipe: EditTemplate) => void;
  disabled?: boolean;
  /** The selected result's run and words, when a model made it. */
  from?: StudioRecipeSource | null;
}) {
  const client = useQueryClient();
  const recipes = useQuery({ queryKey: ["edit-templates"], queryFn: api.editTemplates });
  const [name, setName] = useState("");
  const save = useMutation({
    mutationFn: (source: StudioRecipeSource) =>
      api.createEditTemplate({ name: name.trim(), instruction: source.instruction, from_run_id: source.runId }),
    onSuccess: () => {
      setName("");
      void client.invalidateQueries({ queryKey: ["edit-templates"] });
    },
  });
  const usable = (recipes.data ?? []).filter((recipe) => recipe.enabled);
  if (usable.length === 0 && !from) return null;
  return (
    <div className="studio-recipes">
      <span>
        <strong>Recipes</strong>
      </span>
      {usable.length > 0 && (
        <ul>
          {usable.map((recipe) => (
            <li key={recipe.id}>
              <button
                className="secondary compact-button"
                disabled={disabled}
                title={recipe.description || recipe.instruction}
                onClick={() => onApply(recipe)}
              >
                {recipe.name}
              </button>
              {recipe.mask_mode !== "none" && (
                // Applying it needs a selection, and saying so before the click
                // is the difference between a recipe and a surprise.
                <small>needs a selection</small>
              )}
            </li>
          ))}
        </ul>
      )}
      {from && (
        <form
          className="studio-recipe-save"
          onSubmit={(event) => {
            event.preventDefault();
            if (name.trim() && !save.isPending) save.mutate(from);
          }}
        >
          <label>
            <span>Save this edit as a recipe</span>
            <input value={name} maxLength={200} placeholder="Name" onChange={(event) => setName(event.target.value)} />
          </label>
          <button type="submit" className="secondary compact-button" aria-disabled={!name.trim() || save.isPending}>
            {save.isPending ? "Saving…" : "Save recipe"}
          </button>
          {save.error && <p role="alert">{save.error.message}</p>}
          {save.isSuccess && <small role="status">Saved.</small>}
        </form>
      )}
    </div>
  );
}
