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
  /** A cutout run first, whose subject decides what the words may redraw. */
  cutout?: {
    words: string;
    workflowRevisionId?: string;
    /** Everything around the subject, or the subject itself. */
    redraw: "surroundings" | "subject";
    /** The picture a new subject is taken from, sent after the source. */
    reference?: Blob;
  };
};

export function studioApplyPlan(
  tools: StudioToolState,
  instruction: string,
  recipe: EditTemplate | null,
  activeTool: StudioToolCapability | undefined,
  /** Isolate's entry in the report, which names the workflow every cutout runs on. */
  isolateTool?: StudioToolCapability,
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
  if (tools.kind === "background") {
    const scene = instruction.trim();
    return {
      // The whole picture is redrawn and the subject placed back, so the
      // model is told what goes around the subject and to leave it be.
      words: scene
        ? `Replace the background with ${scene}. Keep the subject exactly as it is, in the same place, size and pose.`
        : "",
      settings: recipe ? recipe.settings_json : undefined,
      workflowRevisionId: recipe?.workflow_revision_id ?? undefined,
      blendSelection: true,
      sendsLightMap: false,
      // The cutout runs the workflow the report names, as Isolate does.
      cutout: {
        words: defaultInstruction({ ...tools, kind: "isolate" }),
        workflowRevisionId: activeTool?.workflow_revision_id ?? undefined,
        redraw: "surroundings",
      },
    };
  }
  if (tools.kind === "subject") {
    const note = instruction.trim();
    return {
      // The model sees both pictures and redraws the first; only the
      // subject's place is kept from what it draws.
      words: `Replace the subject with ${note ? `${note} from` : "the one in"} the second picture. Keep everything around it exactly as it is.`,
      // The report names the first workflow that reads a second picture. The
      // studio's chosen one may read only one, so it is never used here.
      workflowRevisionId: activeTool?.workflow_revision_id ?? undefined,
      blendSelection: true,
      sendsLightMap: false,
      cutout: {
        words: defaultInstruction({ ...tools, kind: "isolate" }),
        workflowRevisionId: isolateTool?.workflow_revision_id ?? undefined,
        redraw: "subject",
        reference: tools.subjectPicture ?? undefined,
      },
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

/** Tools a count does not suit: finding a subject gives the same answer every
 * time, and replacing a background or a subject is two edits, one after the other. */
const ONE_RESULT_TOOLS: readonly string[] = ["isolate", "background", "subject"];

/** Whether an Apply of this model tool can ask for several results. */
export function studioOffersResults(kind: string): boolean {
  return !ONE_RESULT_TOOLS.includes(kind);
}

/** What the apply button says for the tool in hand.
 *
 * The tool's own verb where it has one, "Apply to selection" once a selecting
 * tool has marked something, "Apply edit" otherwise, and "Applying…" while
 * an edit is arriving, whatever the tool.
 */
export function studioApplyLabel(tools: StudioToolState, busy: boolean, selectionCoverage: number): string {
  if (busy) return "Applying…";
  switch (tools.kind) {
    case "extend":
      return "Extend";
    case "text":
      return "Replace words";
    case "relight":
      return "Relight";
    case "isolate":
      return "Cut out";
    case "background":
      return "Replace background";
    case "subject":
      return "Replace subject";
    case "enhance":
      return `Enlarge ${tools.upscaleFactor}x`;
    default:
      return tools.kind !== "instruct" && selectionCoverage > 0 ? "Apply to selection" : "Apply edit";
  }
}
