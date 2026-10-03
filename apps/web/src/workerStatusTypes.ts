import type { CustomNodeContainmentStatus } from "./customNodeContainmentTypes";

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
  startup_duration_ms?: number | null;
  current_memory_bytes: number | null;
  peak_memory_bytes: number | null;
  active_jobs: number;
  queued_jobs: number;
  progress_age_seconds?: number | null;
  failure_detail?: string | null;
  failure_code?:
    | "oom_vram"
    | "oom_host"
    | "port_in_use"
    | "model_incompatible"
    | "executable_missing"
    | "startup_timeout"
    | "crashed"
    | "unknown"
    | null;
  failure_remedy?: string | null;
  stderr_tail?: string | null;
  log_path?: string | null;
  custom_node_containment?: CustomNodeContainmentStatus | null;
}
