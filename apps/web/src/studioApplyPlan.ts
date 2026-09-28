/** What an apply sends for the active tool: the words, the settings, the workflow.
 *
 * Kept out of the studio view so each tool's request reads in one place, and so
 * it can be checked without drawing anything.
 */

import { defaultInstruction, type StudioToolState } from "./studioToolState";
import type { EditTemplate, StudioToolCapability } from "./types";

export type StudioApplyPlan = {
  words: string;
  settings?: Record<string, unknown>;
  workflowRevisionId?: string;
  /** The selection is placed back over the source after the edit, rather than
   * handed to the workflow's own mask input. */
  blendSelection: boolean;
  /** A light map goes with the picture as a second input. */
  sendsLightMap: boolean;
};

export function studioApplyPlan(
  tools: StudioToolState,
  instruction: string,
  recipe: EditTemplate | null,
  activeTool: StudioToolCapability | undefined,
): StudioApplyPlan {
  if (tools.kind === "relight") {
    const adapter = activeTool?.adapter_asset_id;
    return {
      words: defaultInstruction(tools),
      settings: {
        relight: {
          direction: tools.lightDirection,
          intensity: tools.lightIntensity,
          kelvin: tools.lightKelvin,
        },
        // The adapter relights only at full strength; a gentler light is the
        // intensity, which the server mixes in afterwards.
        loras: adapter ? [{ asset_id: adapter, model_strength: 1 }] : [],
      },
      // The report names the workflow when exactly one can relight; otherwise
      // the studio's chosen workflow runs and the server checks it can.
      workflowRevisionId: activeTool?.workflow_revision_id ?? undefined,
      blendSelection: false,
      sendsLightMap: true,
    };
  }
  if (tools.kind === "isolate") {
    return {
      words: defaultInstruction(tools),
      // Nothing to set: the workflow finds the subject and returns only it.
      // It always runs the workflow the report names, never the studio's
      // chosen one, which would answer with an ordinary edit instead.
      workflowRevisionId: activeTool?.workflow_revision_id ?? undefined,
      blendSelection: false,
      sendsLightMap: false,
    };
  }
  // Text takes its words from its own fields. Enhance and Extend ask for no
  // words, and the turn requires some: both were reaching the server and being
  // refused before anything ran. Otherwise the user's words win.
  const words = tools.kind === "text"
    ? defaultInstruction(tools)
    : instruction.trim() || defaultInstruction(tools);
  return {
    words,
    settings: tools.kind === "enhance"
      ? { upscale_factor: tools.upscaleFactor }
      : tools.kind === "extend"
        ? { outpaint_margins: tools.margins }
        : recipe
          ? recipe.settings_json
          : undefined,
    workflowRevisionId: recipe?.workflow_revision_id ?? undefined,
    blendSelection: tools.kind === "text",
    sendsLightMap: false,
  };
}
