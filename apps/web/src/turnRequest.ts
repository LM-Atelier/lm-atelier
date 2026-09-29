import type { ComposerPromptSource } from "./composerPromptSource";
import type { TurnReference } from "./mentionDraft";
import type { SourceFitIntent, SourceFitSelection } from "./sourceFit";
import type { RoutingMode } from "./types";

export interface TurnRequestInput {
  text: string;
  mode: RoutingMode;
  inputArtifactIds: string[];
  settings: Record<string, unknown>;
  idempotencyKey?: string;
  workflowRevisionId?: string;
  references?: TurnReference[];
  outputCount?: number;
  promptSource?: ComposerPromptSource;
  /** An intent asks the server to resolve context; a selection pins a previewed choice. */
  sourceFit?: SourceFitIntent | SourceFitSelection;
}

export interface TurnRequestPayload {
  text: string;
  mode: RoutingMode;
  input_artifact_ids: string[];
  settings: Record<string, unknown>;
  idempotency_key?: string;
  workflow_revision_id?: string;
  references: TurnReference[];
  output_count?: number;
  prompt_source?: ComposerPromptSource;
  source_fit?: SourceFitIntent;
  confirm_media: boolean;
}

export const SOURCE_FIT_BINDING_ERROR = "The source or workflow changed. Preview the canvas again before sending.";

/** Snapshot the actual wire fields before any session initialization or confirmation awaits. */
export function buildTurnRequest(input: TurnRequestInput): TurnRequestPayload {
  const { sourceFit, mode } = input;
  const selection = sourceFit && "request" in sourceFit ? sourceFit : undefined;
  if (sourceFit && (mode !== "image" && mode !== "auto")) throw new Error(SOURCE_FIT_BINDING_ERROR);
  if (selection && (
    !selection.workflowRevisionId
    || input.inputArtifactIds[0] !== selection.sourceArtifactId
    || (input.workflowRevisionId !== undefined && input.workflowRevisionId !== selection.workflowRevisionId)
  )) throw new Error(SOURCE_FIT_BINDING_ERROR);
  const payload: TurnRequestPayload = {
    text: input.text,
    mode,
    input_artifact_ids: input.inputArtifactIds,
    references: input.references ?? [],
    settings: input.settings,
    workflow_revision_id: selection?.workflowRevisionId ?? input.workflowRevisionId,
    source_fit: sourceFit && "request" in sourceFit ? sourceFit.request : sourceFit,
    // Auto must not resurrect a hidden count from a previously selected media mode.
    output_count: mode === "image" || mode === "video" ? input.outputCount : undefined,
    confirm_media: false,
    idempotency_key: input.idempotencyKey,
    prompt_source: input.promptSource,
  };
  return JSON.parse(JSON.stringify(payload)) as TurnRequestPayload;
}
