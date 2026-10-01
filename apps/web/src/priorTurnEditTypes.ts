import type { ComposerPromptSource } from "./composerPromptSource";
import type { SourceFitIntent } from "./sourceFit";
import type { EngineRole, RoutingMode, TurnReferenceInput, TurnRoleOverrides, TurnWorkflowSelectionInput, WorkflowSelection } from "./types";

/** What a prior-turn edit sends, and the configuration it starts from. */

export interface PriorTurnEditRequest {
  source_fit?: SourceFitIntent | null;
  preset_id?: string | null;
  text: string;
  idempotency_key: string;
  source_run_id?: string | null;
  source_snapshot_sha256?: string | null;
  profile_id?: string | null;
  vision_profile_id?: string | null;
  mode?: RoutingMode | null;
  parent_message_id?: string | null;
  /** Omit to inherit source inputs; an empty array explicitly removes them. */
  input_artifact_ids?: string[];
  /** Omit to inherit source bindings; an empty array explicitly removes them. */
  references?: TurnReferenceInput[];
  prompt_source?: ComposerPromptSource | null;
  settings?: Record<string, unknown>;
  ordered_settings?: Record<string, Record<string, unknown>>;
  role_overrides?: Partial<Record<EngineRole, TurnRoleOverrides>>;
  /** Explicit changes to exact source steps; applied after role-wide choices. */
  step_overrides?: Record<string, TurnRoleOverrides>;
  output_count?: number | null;
  workflow_revision_id?: string | null;
  workflow_selection?: TurnWorkflowSelectionInput | null;
  confirm_media?: boolean;
}

export interface PriorTurnEditConfiguration {
  source_fit?: SourceFitIntent | null;
  image_edit_strength?: Record<string, unknown> | null;
  operation: string;
  profile_engine?: string | null;
  settings: Record<string, unknown>;
  resolved_settings: Record<string, unknown>;
  settings_role: string;
  output_count: number;
  profile_id: string | null;
  vision_profile_id: string | null;
  preset_id: string | null;
  preset: Record<string, unknown> | null;
  model_selection: Record<string, unknown>;
  workflow_selection: WorkflowSelection;
  workflow_revision_id: string | null;
  workflow_schema: Record<string, unknown> | null;
  profile_settings?: Record<string, unknown>;
  /** The turn was an enlargement; an edit of it stays one unless it says otherwise. */
  upscale?: boolean;
}
