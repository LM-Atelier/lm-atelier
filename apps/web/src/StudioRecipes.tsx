import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { EditTemplate } from "./types";

/** Saved recipes, offered where the edit is being made.
 *
 * A recipe records the workflow, the model, the settings, and whether the edit
 * expects a selection. Until now the only way to reach one was the chat
 * composer's template dialog, which predates this view entirely - so the place
 * that saves recipes could not use them.
 *
 * A recipe whose workflow is not installed is still shown, and says so. The
 * alternative is hiding it, which reads as "you never saved that" rather than
 * "the thing it needs is missing".
 */
export function StudioRecipes({
  onApply,
  disabled = false,
}: {
  onApply: (recipe: EditTemplate) => void;
  disabled?: boolean;
}) {
  const recipes = useQuery({ queryKey: ["edit-templates"], queryFn: api.editTemplates });
  const usable = (recipes.data ?? []).filter((recipe) => recipe.enabled);
  if (usable.length === 0) return null;
  return (
    <div className="studio-recipes">
      <span>
        <strong>Recipes</strong>
      </span>
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
    </div>
  );
}
