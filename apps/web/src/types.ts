export type RoutingMode = "auto" | "text" | "image" | "video";
export type EngineRole = "chat" | "image" | "video";
export type GenerationSettingsByRole = Partial<Record<EngineRole, Record<string, unknown>>>;
export type GenerationPresetIdsByRole = Partial<Record<EngineRole, string | null>>;

export interface Project {
  id: string;
  name: string;
  description: string;
  instructions: string;
  archived: boolean;
  image_workflow_revision_id: string | null;
  video_workflow_revision_id: string | null;
  generation_settings_json?: GenerationSettingsByRole;
  generation_preset_ids_json?: GenerationPresetIdsByRole;
  created_at: string;
  updated_at: string;
}

export interface Artifact {
  id: string;
  sha256: string;
  kind: string;
  media_type: string;
  size_bytes: number;
  original_name: string | null;
  metadata_json: Record<string, unknown>;
  created_at: string;
  url?: string | null;
}

export interface ArtifactLibraryItem extends Artifact {
  reference_count: number;
  chat_ids: string[];
  project_ids: string[];
}

export interface ArtifactStorageInfo {
  total_bytes: number;
  total_count: number;
  referenced_bytes: number;
  referenced_count: number;
  unreferenced_bytes: number;
  unreferenced_count: number;
  temporary_bytes: number;
  temporary_count: number;
  eligible_bytes: number;
  eligible_count: number;
  retention_pending_count?: number;
  disk_free_bytes: number;
  warning: boolean;
  retention_days: number;
  temporary_retention_hours: number;
}

export interface ArtifactCleanupResult {
  dry_run: boolean;
  marked_count: number;
  retention_pending_count: number;
  removed_count: number;
  reclaimed_bytes: number;
}

export interface ArtifactDeleteResult {
  artifact_id: string;
  reference_count: number;
  removed_count: number;
  reclaimed_bytes: number;
}

export interface MessagePart {
  id: string;
  position: number;
  type: "text" | "image" | "video" | "attachment" | "progress" | "error" | "generation_metadata";
  text: string | null;
  artifact_id: string | null;
  metadata_json: Record<string, unknown>;
  artifact?: Artifact | null;
}

export interface Message {
  id: string;
  chat_id: string;
  parent_id: string | null;
  role: "user" | "assistant" | "system" | "tool";
  status: "complete" | "pending" | "failed" | "cancelled";
  transcript_visible?: boolean;
  active_response_revision_id?: string | null;
  parts: MessagePart[];
  response_revisions?: ResponseRevision[];
  created_at: string;
  updated_at: string;
}

export interface ResponseRevision {
  id: string;
  message_id: string;
  run_id: string | null;
  sequence: number;
  status: "complete" | "pending" | "failed" | "cancelled";
  parts: MessagePart[];
  created_at: string;
  updated_at: string;
}

export interface Chat {
  id: string;
  project_id: string | null;
  title: string;
  archived: boolean;
  routing_mode: RoutingMode;
  confirm_uncertain_media: boolean;
  active_chat_profile_id: string | null;
  active_vision_profile_id?: string | null;
  active_image_profile_id: string | null;
  active_video_profile_id: string | null;
  active_head_message_id: string | null;
  vision_settings_json?: Record<string, unknown>;
  generation_settings_json?: GenerationSettingsByRole;
  generation_preset_ids_json?: GenerationPresetIdsByRole;
  created_at: string;
  updated_at: string;
}

export interface ChatDetail extends Chat {
  messages: Message[];
}

export interface Run {
  id: string;
  idempotency_key: string | null;
  chat_id: string;
  user_message_id: string;
  assistant_message_id: string;
  work_plan_id?: string | null;
  work_step_id?: string | null;
  operation: string;
  status: string;
  standalone_prompt: string;
  profile_id: string | null;
  vision_profile_id?: string | null;
  workflow_revision_id: string | null;
  settings_json: Record<string, unknown>;
  provenance_json: Record<string, unknown>;
  error: string | null;
  created_at: string;
}

export interface TurnAccepted {
  run: Run;
  user_message: Message;
  assistant_message: Message;
}

export interface Job {
  id: string;
  kind: string;
  status: string;
  run_id: string | null;
  work_plan_id?: string | null;
  work_step_id?: string | null;
  progress: number;
  phase: string;
  progress_json?: ProgressV2;
  queue_resource?: string | null;
  queue_group?: string | null;
  queue_priority?: number;
  queue_ticket?: string | null;
  enqueued_at?: string | null;
  claim_expires_at?: string | null;
  heartbeat_at?: string | null;
  payload_json: Record<string, unknown>;
  result_json: Record<string, unknown>;
  error: string | null;
  attempt: number;
  cancellable: boolean;
  created_at: string;
  updated_at: string;
}

export interface ProgressV2 {
  version: 2;
  stage: string;
  stage_progress: number | null;
  overall_progress: number | null;
  completed_units: number | null;
  total_units: number | null;
  unit: string | null;
  bytes_reused: number;
  rate_bytes_per_second: number | null;
  eta_seconds: number | null;
  file_index: number | null;
  file_count: number | null;
  queue_resource: string | null;
  queue_position: number | null;
  queue_length: number | null;
  blocked_by: string[];
  indeterminate: boolean;
  updated_at: string;
}

export interface WorkStep {
  id: string;
  plan_id: string;
  run_id: string | null;
  ordinal: number;
  display_group: string | null;
  operation: string;
  status: string;
  prompt: string;
  profile_id: string | null;
  workflow_revision_id: string | null;
  settings_json: Record<string, unknown>;
  input_bindings_json: Array<Record<string, unknown>>;
  output_contract_json: Array<Record<string, unknown>>;
  queue_class: string;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface WorkPlan {
  id: string;
  chat_id: string;
  idempotency_key: string | null;
  source_action: string;
  persistence_scope: "durable";
  status: string;
  context_head_message_id: string | null;
  transcript_sequence: number;
  priority: number;
  planner_version: string;
  failure_policy: string;
  summary_json: Record<string, unknown>;
  steps: WorkStep[];
  created_at: string;
  updated_at: string;
}

export interface SettingField {
  key: string;
  label: string;
  type: "boolean" | "integer" | "number" | "string" | "enum" | "array" | "object";
  default: unknown;
  minimum: number | null;
  maximum: number | null;
  step: number | null;
  multiple_of?: number | null;
  choices: unknown[];
  scope: "load" | "request" | "workflow";
  visibility: "basic" | "advanced" | "expert";
  restart_required: boolean;
  available: boolean;
  unavailable_reason: string | null;
  help: string;
}

export interface EngineCapabilities {
  engine: string;
  version: string;
  roles: string[];
  operations: string[];
  formats: string[];
  devices: string[];
  input_modalities?: string[];
  streaming: boolean;
  tool_calling: boolean;
  settings: SettingField[];
  settings_by_role?: Partial<Record<EngineRole, SettingField[]>>;
  healthy: boolean;
  details: Record<string, unknown>;
}

export interface ToolCapabilityProbe {
  engine: string;
  version: string;
  advertised: boolean;
  passed: boolean;
  tool_name: string | null;
  arguments: Record<string, unknown> | null;
  error: string | null;
}

export interface ModelProfile {
  id: string;
  model_install_id: string | null;
  name: string;
  use_case: string;
  role: "chat" | "image" | "video";
  engine: string;
  load_settings_json: Record<string, unknown>;
  request_settings_json: Record<string, unknown>;
  is_default: boolean;
  input_modalities?: string[];
}

export interface ModelProfileBundle {
  format: "lm-atelier-profile";
  version: 1;
  name: string;
  use_case: string;
  role: "chat" | "image" | "video";
  engine: string;
  model_install_id: string | null;
  load_settings: Record<string, unknown>;
  request_settings: Record<string, unknown>;
}

export interface GenerationPreset {
  id: string;
  name: string;
  role: "chat" | "image" | "video";
  settings_json: Record<string, unknown>;
  is_default: boolean;
}

export interface GenerationPresetBundle {
  format: "lm-atelier-preset";
  version: 1;
  name: string;
  role: "chat" | "image" | "video";
  settings: Record<string, unknown>;
}

export interface WorkerStatus {
  name: "chat" | "media";
  state: "stopped" | "starting" | "ready" | "exited";
  managed: boolean;
  running: boolean;
  pid: number | null;
  profile_id: string | null;
  command: string[];
  exit_code: number | null;
  estimated_memory_bytes: number | null;
  current_memory_bytes: number | null;
  peak_memory_bytes: number | null;
  active_jobs: number;
  queued_jobs: number;
  failure_detail?: string | null;
  stderr_tail?: string | null;
  log_path?: string | null;
}

export interface RuntimeStatus {
  engine: "llama.cpp" | "vllm" | "comfyui";
  release: string;
  state: "missing" | "installing" | "ready" | "failed" | "unsupported";
  supported: boolean;
  managed: boolean;
  progress: number;
  progress_json?: ProgressV2 | null;
  downloaded_bytes: number;
  size_bytes: number | null;
  distribution: string;
  license: string;
  security_status?: "checksum-pinned" | "blocked";
  security_message?: string;
  message: string;
}

export interface BackupInfo {
  name: string;
  size_bytes: number;
  sha256: string;
  created_at: string;
  verified: boolean;
  restore_pending: boolean;
  media_included: boolean;
  media_size_bytes: number;
}

export interface ModelInstall {
  id: string;
  source_id: string | null;
  name: string;
  role: string;
  engine: string;
  local_path: string;
  size_bytes: number;
  compatibility: string;
  manifest_json: Record<string, unknown>;
  active: boolean;
  readiness: "ready" | "unverified" | "unsupported";
  capability_evidence: {
    id: string;
    evidence_key: string;
    result: string;
    runtime_build: string;
    probed_at: string;
  } | null;
  created_at: string;
  updated_at: string;
}

export interface ModelAssetInstall {
  id: string;
  source_id: string | null;
  name: string;
  kind: string;
  family: string | null;
  size_bytes: number;
  manifest_json: Record<string, unknown>;
  active: boolean;
  verified_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface ModelStorageInfo {
  installed_bytes: number;
  partial_download_bytes: number;
  catalog_cache_bytes: number;
  installed_count: number;
  partial_download_count: number;
}

export interface StorageCleanupResult {
  removed_count: number;
  reclaimed_bytes: number;
}

export interface CatalogModel {
  provider: string;
  remote_id: string;
  name: string;
  author: string | null;
  pipeline_tag: string | null;
  tags: string[];
  downloads: number | null;
  likes: number | null;
  trending_score: number | null;
  created_at: string | null;
  last_modified: string | null;
  gated: boolean | string | null;
  private: boolean;
  library_name: string | null;
  architecture: string | null;
  formats: string[];
  quantizations: string[];
  parameter_count: number | null;
  license_id: string | null;
  total_size_bytes: number | null;
  compatibility: string;
  compatibility_reasons: string[];
  required_runtime?: string | null;
}

export interface CatalogPage {
  items: CatalogModel[];
  next_cursor: string | null;
  stale?: boolean;
}

export interface CatalogDetail {
  model: CatalogModel;
  revision: string;
  files: Array<{ filename: string; size: number | null; sha256: string | null }>;
}

export interface CatalogPreflight {
  remote_id: string;
  source_remote_id: string | null;
  revision: string;
  selected_files: string[];
  expected_sha256: Record<string, string>;
  file_sources?: Record<string, {
    remote_id: string;
    revision: string;
    filename: string;
    size_bytes: number | null;
    sha256: string | null;
  }>;
  comfy_paths: Record<string, string>;
  workflow_template_id: string | null;
  workflow_template_sha256: string | null;
  download_bytes: number;
  available_disk_bytes: number;
  estimated_ram_bytes: number | null;
  estimated_vram_bytes: number | null;
  can_install: boolean;
  auxiliary_kind?: string | null;
  install_plan: {
    id: string;
    plan_hash: string;
    compatibility: "supported" | "unsupported" | "trusted_extension_required";
    family: string | null;
    failure_code: string | null;
    failure_reason: string | null;
  } | null;
  checks: Array<{
    id: string;
    label: string;
    status: "pass" | "warn" | "block";
    detail: string;
  }>;
}

export interface RecipeFile {
  path: string;
  size_bytes: number | null;
  sha256: string | null;
}

export interface ReferenceRecipe {
  id: string;
  version: number;
  name: string;
  summary: string;
  role: "chat" | "image" | "video";
  engine: "llama.cpp" | "vllm" | "comfyui";
  operations: string[];
  license_id: string;
  status: "reference-candidate" | "certified";
  certified: boolean;
  remote_id: string;
  revision: string;
  files: RecipeFile[];
  total_size_bytes: number | null;
  hardware: {
    tier: "cpu" | "midrange-gpu" | "high-end-gpu";
    minimum_ram_gb: number;
    recommended_ram_gb: number;
    minimum_vram_gb: number | null;
    recommended_vram_gb: number | null;
    guidance: string;
  };
  default_settings: Record<string, unknown>;
  workflow_path: string | null;
  node_policy: string | null;
  notes: string[];
}

export interface WorkflowRevision {
  id: string;
  workflow_id: string;
  version: number;
  engine: string;
  engine_version: string | null;
  ui_graph_json: Record<string, unknown>;
  api_graph_json: Record<string, unknown>;
  input_schema_json: Record<string, unknown>;
  dependencies_json: Record<string, unknown>;
  trusted: boolean;
  created_at: string;
}

export interface Workflow {
  id: string;
  name: string;
  operation: string;
  description: string;
  current_revision_id: string | null;
  revisions: WorkflowRevision[];
}

export interface CustomNodeInstall {
  id: string;
  name: string;
  source_url: string;
  revision: string;
  previous_revision: string | null;
  tree_hash: string;
  trusted: boolean;
  active: boolean;
  security_json: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface WorkflowBundle {
  format: "lm-atelier-workflow";
  version: 1;
  name: string;
  operation: string;
  description: string;
  engine: string;
  engine_version: string | null;
  ui_graph: Record<string, unknown>;
  api_graph: Record<string, unknown>;
  input_schema: Record<string, unknown>;
  dependencies: Record<string, unknown>;
  trusted: boolean;
  source_revision: number | null;
}

export interface SystemInfo {
  platform: string;
  platform_release: string;
  distribution: string;
  distribution_version: string;
  architecture: string;
  python_version: string;
  cpu_model: string;
  cpu_count: number;
  memory_total_bytes: number;
  memory_available_bytes: number;
  disk_total_bytes: number;
  disk_free_bytes: number;
  ffmpeg_available: boolean;
  support: PlatformAssessment;
  devices: Array<{
    id: string;
    name: string;
    kind: string;
    total_memory_bytes: number | null;
    available_memory_bytes: number | null;
    backend: string | null;
    details: Record<string, unknown>;
  }>;
}

export interface ApplicationInfo {
  version: string;
  data_directory: string;
  log_directory: string;
}

export interface CredentialStatus {
  provider: "huggingface";
  configured: boolean;
  source: "none" | "environment" | "credential_vault";
  vault_available: boolean;
}

export interface PlatformAssessment {
  platform_status: "target" | "experimental" | "unsupported";
  platform_label: string;
  accelerator_status: "primary" | "experimental" | "cpu-only";
  accelerator_label: string;
  certification_status: "hardware-pending" | "experimental" | "unsupported";
  chat_ready: boolean;
  reference_media_ready: boolean;
  vram_tier_gb: number | null;
  messages: string[];
}

export interface PlatformMatrixEntry {
  id: string;
  name: string;
  status: "target" | "experimental";
  operating_systems: string[];
  architectures: string[];
  accelerator: string;
  workloads: string[];
  vram_tiers_gb: number[];
  evidence: string;
  notes: string[];
}

export interface AppEvent {
  sequence: number;
  type: string;
  entity_id: string | null;
  payload: Record<string, unknown>;
  created_at: string;
}
