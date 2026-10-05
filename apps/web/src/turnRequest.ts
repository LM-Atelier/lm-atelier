import type { ComposerPromptSource } from "./composerPromptSource";
import type { TurnReference } from "./mentionDraft";
import type { SourceFitIntent, SourceFitSelection } from "./sourceFit";
import type { ImageInputRole, OutputRatioPresetId, RoutingMode } from "./types";

/** The shape new pictures and videos take when nothing else sets their size. */
export type DefaultOutputShapes = { image: OutputRatioPresetId | null; video: OutputRatioPresetId | null };

export interface TurnRequestInput {
  text: string;
  mode: RoutingMode;
  inputArtifactIds: string[];
  inputImageRoles?: ImageInputRole[];
  settings: Record<string, unknown>;
  idempotencyKey?: string;
  workflowRevisionId?: string;
  references?: TurnReference[];
  outputCount?: number;
  promptSource?: ComposerPromptSource;
  /** An intent asks the server to resolve context; a selection pins a previewed choice. */
  sourceFit?: SourceFitIntent | SourceFitSelection;
  /** Sent only when one is chosen, so a turn without any reads as it always did. */
  defaultOutputShapes?: DefaultOutputShapes;
  /** An enlargement, said as one rather than read from a scale factor that a
   * workflow which sets its own size does not take. */
  upscale?: boolean;
}

export interface TurnRequestPayload {
  text: string;
  mode: RoutingMode;
  input_artifact_ids: string[];
  input_image_roles?: ImageInputRole[];
  settings: Record<string, unknown>;
  idempotency_key?: string;
  workflow_revision_id?: string;
  references: TurnReference[];
  output_count?: number;
  prompt_source?: ComposerPromptSource;
  source_fit?: SourceFitIntent;
  default_output_shapes?: DefaultOutputShapes;
  upscale?: true;
  confirm_media: boolean;
}

export const SOURCE_FIT_BINDING_ERROR = "The source or workflow changed. Preview the canvas again before sending.";

/** Snapshot the actual wire fields before any session initialization or confirmation awaits. */
export function buildTurnRequest(input: TurnRequestInput): TurnRequestPayload {
  const { sourceFit, mode, defaultOutputShapes: shapes } = input;
  const selection = sourceFit && "request" in sourceFit ? sourceFit : undefined;
  const roles = input.inputImageRoles;
  if (roles && (roles.length !== input.inputArtifactIds.length || new Set(input.inputArtifactIds).size !== roles.length
    || roles.some((role) => role !== "edit_source" && role !== "reference")
    || roles.filter((role) => role === "edit_source").length > 1)) throw new Error("Choose a purpose for each picture before sending.");
  const sourceIndex = roles ? roles.indexOf("edit_source") : 0;
  if (sourceFit && (mode !== "image" && mode !== "auto")) throw new Error(SOURCE_FIT_BINDING_ERROR);
  if (selection && (
    !selection.workflowRevisionId
    || input.inputArtifactIds[sourceIndex] !== selection.sourceArtifactId
    || (input.workflowRevisionId !== undefined && input.workflowRevisionId !== selection.workflowRevisionId)
  )) throw new Error(SOURCE_FIT_BINDING_ERROR);
  const payload: TurnRequestPayload = {
    text: input.text,
    mode,
    input_artifact_ids: input.inputArtifactIds,
    ...(roles ? { input_image_roles: roles } : {}),
    references: input.references ?? [],
    settings: input.settings,
    workflow_revision_id: selection?.workflowRevisionId ?? input.workflowRevisionId,
    source_fit: sourceFit && "request" in sourceFit ? sourceFit.request : sourceFit,
    // Auto must not resurrect a hidden count from a previously selected media mode.
    output_count: mode === "image" || mode === "video" ? input.outputCount : undefined,
    confirm_media: false,
    idempotency_key: input.idempotencyKey,
    prompt_source: input.promptSource,
    default_output_shapes: shapes && (shapes.image || shapes.video) ? shapes : undefined,
    // Only an enlargement says so; every other turn reads as it always did.
    upscale: input.upscale ? true : undefined,
  };
  return JSON.parse(JSON.stringify(payload)) as TurnRequestPayload;
}
