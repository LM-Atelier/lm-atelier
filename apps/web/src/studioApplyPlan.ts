/** What an apply sends for the active tool: the words, the settings, the workflow.
 *
 * Kept out of the studio view so each tool's request reads in one place, and so
 * it can be checked without drawing anything.
 */

import { chosenFactor, enlargementFactor, type EnlargementPreview } from "./studioEnlargement";
import { defaultInstruction, type StudioSubjectPicture, type StudioToolState } from "./studioToolState";
import type { EditTemplate, StudioToolCapability } from "./types";

export type StudioApplyPlan = {
  words: string;
  settings?: Record<string, unknown>;
  workflowRevisionId?: string;
  /** The selection is placed back over the source after the edit, rather than
   * handed to the workflow's own mask input. */
  blendSelection: boolean;
  /** The selection reaches past what was marked before it is softened, so the
   * edge of what is removed is redrawn too. */
  growSelection?: boolean;
  /** A light map goes with the picture as a second input. */
  sendsLightMap: boolean;
  /** Said as an enlargement, so a workflow that sets its own size is still chosen as one. */
  upscale?: boolean;
  /** A cutout run first, whose subject decides what the words may redraw. */
  cutout?: {
    words: string;
    workflowRevisionId?: string;
    /** Everything around the subject, or the subject itself. */
    redraw: "surroundings" | "subject";
    /** The picture a new subject is cut out of: its bytes, or the artifact
     * when the library already holds it. */
    reference?: Blob | string;
  };
};

/** Where a new subject is cut out from: a chosen file's bytes, or the artifact
 * of one the library already holds, which is not sent again. */
function subjectReference(picture: StudioSubjectPicture | null): Blob | string | undefined {
  if (!picture) return undefined;
  return picture instanceof File ? picture : picture.artifactId;
}

export function studioApplyPlan(
  tools: StudioToolState,
  instruction: string,
  recipe: EditTemplate | null,
  activeTool: StudioToolCapability | undefined,
  /** Isolate's entry in the report, which names the workflow every cutout runs on. */
  isolateTool?: StudioToolCapability,
  /** Which workflow an Enhance runs and what it takes, as the server previewed it. */
  enlargement?: EnlargementPreview | null,
): StudioApplyPlan {
  if (tools.kind === "enhance") {
    const factor = enlargement
      ? enlargementFactor(enlargement.factor, chosenFactor(enlargement, tools.upscaleChoice))
      : null;
    return {
      words: defaultInstruction(tools),
      // Only a factor the workflow applies is sent; one that sets its own size is given none.
      settings: factor !== null ? { upscale_factor: factor } : undefined,
      // The workflow the preview named, so the run is the one the person was shown.
      workflowRevisionId: enlargement?.workflow_revision_id,
      upscale: true,
      blendSelection: false,
      sendsLightMap: false,
    };
  }
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
    return {
      // The old subject is removed as Remove removes a marked part: the model
      // redraws the picture without it, on the studio's own workflow, and only
      // its grown outline is kept. The new one is then placed where it stood,
      // so where it goes is never the model's choice.
      words: "Remove the subject. Fill the space it leaves to match what surrounds it, and leave everything else unchanged.",
      blendSelection: true,
      sendsLightMap: false,
      // Both cutouts run the workflow the report names for Isolate.
      cutout: {
        words: defaultInstruction({ ...tools, kind: "isolate" }),
        workflowRevisionId: isolateTool?.workflow_revision_id ?? undefined,
        redraw: "subject",
        reference: subjectReference(tools.subjectPicture),
      },
    };
  }
  if (tools.kind === "remove") {
    const target = instruction.trim();
    return {
      // The model redraws the whole picture without it and only the marked part
      // is kept, so it is told what goes and to leave the rest as it was.
      words: target
        ? `Remove ${target}. Fill the space it leaves to match what surrounds it, and leave everything else unchanged.`
        : "",
      blendSelection: true,
      growSelection: true,
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
    settings: tools.kind === "extend"
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

/** Tools that ask for no words of their own. */
const WORDLESS_TOOLS: readonly string[] = ["enhance", "extend", "text", "relight", "isolate", "subject"];
/** Tools that run on the workflow the report names for them, whatever the chat has chosen. */
const OWN_WORKFLOW_TOOLS: readonly string[] = ["relight", "isolate"];

/** Whether the tool in hand has everything its edit needs.
 *
 * Enhance asks for no words: the whole picture is the subject and the size is
 * the whole instruction. Text takes its words from its own fields, and without
 * a box it would change the whole picture; Remove, too, needs a marked part as
 * well as its words. Isolate asks for nothing and runs only the workflow the
 * report names. Replacing a subject needs only the picture it comes from: both
 * subjects are cut out on the workflow the report names, and the old one is
 * removed on the studio's own, as a replaced background's surroundings are
 * redrawn on it.
 */
export function studioToolReady(
  tools: StudioToolState,
  instruction: string,
  recipe: EditTemplate | null,
  selectionCoverage: number,
  activeTool: StudioToolCapability | undefined,
  isolateTool: StudioToolCapability | undefined,
  /** Why the chat's own workflow choice cannot run, if it cannot. */
  workflowUnavailable: string | null,
  /** Which workflow an Enhance runs, once the server has said. */
  enlargement?: EnlargementPreview | null,
): boolean {
  // An enlargement runs the workflow the preview names, so it waits for one.
  if (tools.kind === "enhance") return Boolean(enlargement);
  const ownWorkflow = Boolean(activeTool?.workflow_revision_id);
  if (tools.kind === "extend" && !Object.values(tools.margins).some(Boolean)) return false;
  if (tools.kind === "text" && (!tools.newWords.trim() || selectionCoverage === 0)) return false;
  if (tools.kind === "remove" && selectionCoverage === 0) return false;
  if ((tools.kind === "isolate" || tools.kind === "background") && !ownWorkflow) return false;
  if (
    tools.kind === "subject"
    && (!ownWorkflow || !isolateTool?.workflow_revision_id || !tools.subjectPicture)
  ) {
    return false;
  }
  if (!WORDLESS_TOOLS.includes(tools.kind) && !instruction.trim()) return false;
  if (recipe !== null && recipe.mask_mode !== "none" && selectionCoverage === 0) return false;
  return !workflowUnavailable || Boolean(recipe?.workflow_revision_id)
    || (OWN_WORKFLOW_TOOLS.includes(tools.kind) && ownWorkflow);
}

/** What the apply button says for the tool in hand.
 *
 * The tool's own verb where it has one, "Apply to selection" once a selecting
 * tool has marked something, "Apply edit" otherwise, and "Applying…" while
 * an edit is arriving, whatever the tool.
 */
export function studioApplyLabel(
  tools: StudioToolState,
  busy: boolean,
  selectionCoverage: number,
  enlargement?: EnlargementPreview | null,
): string {
  if (busy) return "Applying…";
  switch (tools.kind) {
    case "extend":
      return "Extend";
    case "text":
      return "Replace words";
    case "remove":
      return "Remove";
    case "relight":
      return "Relight";
    case "isolate":
      return "Cut out";
    case "background":
      return "Replace background";
    case "subject":
      return "Replace subject";
    case "enhance":
      // Only a size the workflow's graph proves is named. A factor the person
      // chooses sets the workflow's scale, which another step may multiply.
      return enlargement && enlargement.fixed_factor !== null ? `Enlarge ${enlargement.fixed_factor}x` : "Enlarge";
    default:
      return tools.kind !== "instruct" && selectionCoverage > 0 ? "Apply to selection" : "Apply edit";
  }
}
