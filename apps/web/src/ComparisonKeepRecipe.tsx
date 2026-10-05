import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api";
import { GenerationEditRecipeDialog } from "./GenerationEditRecipeDialog";
import { comparisonFailure } from "./generationComparison";
import type { ExperimentArm, ExperimentOperation } from "./generationExperimentTypes";
import { RecipeDraftDialog } from "./RecipeDraftDialog";

/** Keep the setup one choice ran with: its settings as a recipe, reviewed before it is saved.
 *
 * A recipe holds settings only, so the model and workflow are named beside it,
 * and a new chat can be set up with all three. A choice that changed a picture
 * is kept the way any edit is, as an Image Studio recipe read from the run that
 * made its picture. Nothing here changes a default or what Automatic chooses.
 */
export function ComparisonKeepRecipe({ experimentId, operation, arm, onOpenChat }: {
  experimentId: string;
  operation: ExperimentOperation;
  arm: ExperimentArm;
  /** Show a chat; without it the recipe is saved and no chat is offered. */
  onOpenChat?: (chatId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const made = arm.trials.find((trial) => trial.status === "complete")?.run_id ?? null;
  if (operation === "image_to_image" && !made) return null;
  return <>
    <button type="button" className="secondary compact-button" onClick={() => setOpen(true)}>Keep as a recipe</button>
    {open && (operation === "image_to_image" && made
      ? <GenerationEditRecipeDialog runId={made} onClose={() => setOpen(false)} />
      : <ComparisonRecipeDialog experimentId={experimentId} arm={arm} onOpenChat={onOpenChat}
        onClose={() => setOpen(false)} />)}
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
  return <RecipeDraftDialog title={`Keep ${arm.label} as a recipe`} eyebrow="Recipe from a comparison" draft={draft}
    failure={(error) => comparisonFailure(error).message} onOpenChat={onOpenChat} onClose={onClose} />;
}
