import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { comparisonFailure } from "./generationComparison";
import type { ExperimentEvaluation, GenerationExperiment, GenerationExperimentEvaluationCreate } from "./generationExperimentTypes";
import "./GenerationComparisonView.css";

/** Say which picture is preferred, that they tie, or that neither suits.
 *
 * Each saying is kept, and the latest is shown as the comparison's answer. It
 * is the person's own judgement: nothing here calls a picture better.
 */
export function ComparisonPreference({ experiment }: { experiment: GenerationExperiment }) {
  const client = useQueryClient();
  const evaluate = useMutation({
    mutationFn: (said: GenerationExperimentEvaluationCreate) => api.evaluateGenerationExperiment(experiment.id, said),
    onSuccess: (updated) => client.setQueryData(["generation-experiments", experiment.id], updated),
  });
  const said = experiment.evaluation;
  const say = (next: GenerationExperimentEvaluationCreate) => {
    if (!evaluate.isPending) evaluate.mutate(next);
  };
  return <fieldset className="comparison-preference">
    <legend>Which do you prefer?</legend>
    <div className="row-actions">
      {experiment.arms.map((arm) => {
        // A choice is preferred only once its own picture is made.
        const unmade = !arm.trials.some((trial) => trial.status === "complete");
        return <button key={arm.id} type="button" className="secondary" aria-disabled={evaluate.isPending || unmade}
          aria-pressed={said?.preference === "preferred" && said.arm_ordinal === arm.ordinal}
          onClick={() => { if (!unmade) say({ preference: "preferred", arm_ordinal: arm.ordinal }); }}>Prefer {arm.label}</button>;
      })}
      <button type="button" className="secondary" aria-disabled={evaluate.isPending} aria-pressed={said?.preference === "tied"}
        onClick={() => say({ preference: "tied" })}>Tie</button>
      <button type="button" className="secondary" aria-disabled={evaluate.isPending} aria-pressed={said?.preference === "unsuitable"}
        onClick={() => say({ preference: "unsuitable" })}>Neither suits</button>
    </div>
    {said && <p role="status">{saying(said, experiment)}</p>}
    <ErrorCallout message={evaluate.error ? comparisonFailure(evaluate.error).message : null} />
  </fieldset>;
}

function saying(said: ExperimentEvaluation, experiment: GenerationExperiment): string {
  if (said.preference === "tied") return "You called it a tie.";
  if (said.preference === "unsuitable") return "You said neither suits.";
  const label = experiment.arms.find((arm) => arm.ordinal === said.arm_ordinal)?.label ?? "one choice";
  return `You preferred ${label}.`;
}
