import { useQueries } from "@tanstack/react-query";
import { api } from "./api";
import { arrangedShapes, useOutputShapes } from "./outputShapePreferences";
import type { ShapeAlternatives } from "./shapeAlternatives";
import { WorkflowFamilyBrowseControls } from "./WorkflowFamilyBrowseControls";

/** The workflows among the alternatives that take a shape, each one choice away.
 *
 * Each candidate is asked the same question the shape row asks of the chosen
 * workflow, so a workflow is offered only where choosing it would bring up
 * shapes that can actually be chosen: at least one shape it proves, and not
 * every one of them left out in Settings.
 */
export function ShapeAlternativeList({ alternatives }: { alternatives: ShapeAlternatives }) {
  const shapes = useOutputShapes();
  const proofs = useQueries({
    queries: alternatives.candidates.map((candidate) => ({
      queryKey: ["workflow-revision", candidate.revisionId, "output-geometry"],
      queryFn: () => api.workflowRevisionOutputGeometry(candidate.revisionId),
    })),
  });
  const shaped = alternatives.candidates.filter((_candidate, index) => {
    const proof = proofs[index]?.data;
    if (!proof?.available) return false;
    const mode = proof.operation === "text_to_image" ? "image" : "video";
    return arrangedShapes(proof.preset_ids, shapes[mode]).length > 0;
  });
  if (shaped.length === 0 && !alternatives.browse) return null;
  return (
    <div>
      {alternatives.browse && <WorkflowFamilyBrowseControls browse={alternatives.browse} label="alternative workflows" />}
      {shaped.length > 0 && <small>{"These workflows take a shape: "}
      {shaped.map((candidate, index) => (
        <span key={candidate.familyId}>
          {index > 0 && ", "}
          <button
            type="button"
            className="link-button"
            // Not disabled: a focused button that becomes disabled drops focus to the page.
            aria-disabled={alternatives.choosing}
            onClick={() => {
              if (!alternatives.choosing) alternatives.onChoose(candidate.familyId);
            }}
          >
            {`Use ${candidate.name}`}
          </button>
        </span>
      ))}
      </small>}
      {alternatives.error && <span role="alert">{` ${alternatives.error}`}</span>}
    </div>
  );
}
