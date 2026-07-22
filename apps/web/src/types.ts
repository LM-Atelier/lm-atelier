export type RoutingMode = "auto" | "text" | "image" | "video";

export interface Project {
  id: string;
  name: string;
  description: string;
  instructions: string;
  archived: boolean;
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

export interface MessagePart {
  id: string;
  position: number;
  type: "text" | "image" | "video" | "progress" | "error" | "generation_metadata";
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
  active_image_profile_id: string | null;
  active_video_profile_id: string | null;
  active_head_message_id: string | null;
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
  operation: string;
  status: string;
  standalone_prompt: string;
  profile_id: string | null;
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
  progress: number;
  phase: string;
  payload_json: Record<string, unknown>;
  result_json: Record<string, unknown>;
  error: string | null;
  attempt: number;
  cancellable: boolean;
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
  streaming: boolean;
  tool_calling: boolean;
  settings: SettingField[];
  healthy: boolean;
  details: Record<string, unknown>;
}

export interface ModelProfile {
  id: string;
  model_install_id: string | null;
  name: string;
  role: "chat" | "image" | "video";
  engine: string;
  load_settings_json: Record<string, unknown>;
  request_settings_json: Record<string, unknown>;
  is_default: boolean;
}

export interface WorkerStatus {
  name: "chat" | "media";
  managed: boolean;
  running: boolean;
  pid: number | null;
  profile_id: string | null;
  command: string[];
  exit_code: number | null;
}

export interface BackupInfo {
  name: string;
  size_bytes: number;
  sha256: string;
  created_at: string;
  verified: boolean;
  restore_pending: boolean;
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
  created_at: string;
  updated_at: string;
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
  compatibility: string;
  compatibility_reasons: string[];
}

export interface CatalogPage {
  items: CatalogModel[];
  next_cursor: string | null;
}

export interface CatalogDetail {
  model: CatalogModel;
  files: Array<{ filename: string; size: number | null; sha256: string | null }>;
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
  engine: "llama.cpp" | "comfyui";
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

export interface SystemInfo {
  platform: string;
  platform_release: string;
  distribution: string;
  distribution_version: string;
  architecture: string;
  python_version: string;
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
