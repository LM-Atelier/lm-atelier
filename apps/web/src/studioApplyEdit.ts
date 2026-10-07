/** What pressing Apply does: prepare what the edit sends with the picture, then send it.
 *
 * Kept out of the studio view, which decides only whether Apply can be pressed,
 * so each way an edit leaves the browser reads in one place.
 */

import { studioApplyPlan, studioOffersResults } from "./studioApplyPlan";
import { subjectReach } from "./studioBackground";
import { renderLightMap } from "./studioLightMap";
import { cloneMask, dilate, encodeMaskPng, feather, isEmpty, type MaskRaster } from "./studioMasks";
import type { EnlargementPreview } from "./studioEnlargement";
import { toolUsesMask, type StudioToolState } from "./studioToolState";
import type { EditTemplate, StudioToolCapability, TurnAccepted } from "./types";
import type { useStudioBackground } from "./useStudioBackground";
import type { StudioStep, useStudioSession } from "./useStudioSession";
import type { useStudioSubjectPlace } from "./useStudioSubjectPlace";

export const SELECTION_NOT_PREPARED =
  "The selection could not be prepared, so nothing was sent. Try again, or clear the selection to edit the whole picture.";
export const LIGHT_MAP_NOT_PREPARED =
  "The light map could not be drawn in this browser, so nothing was sent.";

export type StudioApplyEdit = {
  tools: StudioToolState;
  instruction: string;
  recipe: EditTemplate | null;
  activeTool: StudioToolCapability | undefined;
  /** Isolate's entry in the report, which names the workflow every cutout runs on. */
  isolateTool: StudioToolCapability | undefined;
  /** The picture on the canvas, which the edit changes. */
  current: StudioStep;
  /** Its decoded pixels: a light map or a cutout is drawn at their size. */
  bitmap: ImageBitmap | null;
  results: number;
  apply: ReturnType<typeof useStudioSession>["apply"];
  cutout: Pick<ReturnType<typeof useStudioBackground>, "start">;
  /** Replacing a subject, which takes steps of its own. */
  subject: Pick<ReturnType<typeof useStudioSubjectPlace>, "start">;
  /** Says why nothing was sent, or with null that nothing is wrong any longer. */
  setError: (message: string | null) => void;
  /** Runs once the edit is taken, with its turn when it was sent as one: a replaced
   * background or subject takes several steps, and says only that all were taken. */
  onAccepted: (accepted?: TurnAccepted) => void;
  /** Which workflow an Enhance runs and what it takes, as the server previewed it. */
  enlargement?: EnlargementPreview | null;
  /** Runs when the turn is refused, so a preview it was built on can be asked again. */
  onRefused?: () => void;
};

export function applyStudioEdit(edit: StudioApplyEdit): void {
  const { tools, recipe, current, bitmap } = edit;
  const selection = toolUsesMask(tools.kind) && tools.mask && !isEmpty(tools.mask)
    ? tools.mask
    : null;
  const plan = studioApplyPlan(tools, edit.instruction, recipe, edit.activeTool, edit.isolateTool, edit.enlargement);
  if (plan.cutout) {
    // Drawn at the picture's own size, so the subject lines up with it.
    if (!bitmap) return;
    edit.setError(null);
    const replaces = plan.cutout.redraw === "subject" ? edit.subject : edit.cutout;
    replaces.start(plan, current.artifactId, { width: bitmap.width, height: bitmap.height }, edit.onAccepted);
    return;
  }
  const send = (mask: Blob | null, secondPicture?: Blob) => {
    edit.apply(
      plan.words,
      current.artifactId,
      mask
        ? {
            blob: mask,
            featherPx: tools.featherPx,
            // A recipe made on everything outside a selection changes that again.
            invert: recipe?.mask_mode === "inverse",
            ...(plan.blendSelection ? { apply: "blend" as const } : {}),
          }
        : undefined,
      plan.settings,
      plan.workflowRevisionId,
      edit.onAccepted,
      secondPicture,
      edit.onRefused,
      studioOffersResults(tools.kind) ? edit.results : 1,
      plan.upscale,
    );
  };
  edit.setError(null);
  if (plan.sendsLightMap) {
    // Drawn at the picture's own size, so the map and the picture line up.
    if (!bitmap) return;
    // Drawing can throw as well as come back empty; both refuse the same way.
    void renderLightMap(bitmap.width, bitmap.height, tools.lightDirection).then(
      (map) => (map ? send(null, map) : edit.setError(LIGHT_MAP_NOT_PREPARED)),
      () => edit.setError(LIGHT_MAP_NOT_PREPARED),
    );
  } else if (selection) {
    // A selection that cannot be encoded is refused, never sent as an
    // edit of the whole picture it was drawn to protect.
    const grow = plan.growSelection ? subjectReach(selection.width, selection.height) : 0;
    void encodeMaskPng(plan.blendSelection ? softened(selection, tools.featherPx, grow) : selection).then(
      (mask) => (mask ? send(mask) : edit.setError(SELECTION_NOT_PREPARED)),
      () => edit.setError(SELECTION_NOT_PREPARED),
    );
  } else send(null);
}

/** A copy of the selection with softened edges, leaving the one on the canvas as drawn.
 *
 * Text is placed back through its box, and a hard edge would show wherever the
 * edited picture differs slightly from the source just outside the words. What
 * Remove takes out is grown first, by the reach a replaced subject is given:
 * a selection held to the old outline keeps a ring of what was removed.
 */
function softened(mask: MaskRaster, featherPx: number, growPx = 0): MaskRaster {
  const copy = cloneMask(mask);
  if (growPx > 0) dilate(copy, growPx);
  if (featherPx > 0) feather(copy, featherPx);
  return copy;
}
