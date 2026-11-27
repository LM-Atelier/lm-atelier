import type { CustomNodeContainmentStatus } from "./customNodeContainmentTypes";

/** Why a worker process was started. */
export type WorkerStartReason =
  | "app_start"
  | "run"
  | "handoff_restart"
  | "manual"
  | "package_preparation"
  | "model_change"
  | "unattributed";

/**
 * Why a running worker process was stopped. `failed_run_handoff` is a recycle
 * after a run that did not finish with its result and was not allowed to keep
 * its worker; `run_handoff` is a recycle at a run's own handoff other than a
 * failed run's.
 */
export type WorkerStopReason =
  | "run_handoff"
  | "failed_run_handoff"
  | "keep_warm_expiry"
  | "chat_load"
  | "replaced_scope"
  | "replaced_command"
  | "manual"
  | "package_preparation"
  | "model_change"
  | "unhealthy"
  | "shutdown"
  | "unattributed";

export interface WorkerStatus {
  name: "chat" | "media";
  state: "stopped" | "starting" | "ready" | "stopping" | "exited";
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
  // Counted since `counted_since`, when the service started. A process that
  // ended while nothing was stopping it is an exit without a stop, not a stop.
  // An emptying in place stood in for a stop with that reason; the process ran on.
  starts_by_reason?: Partial<Record<WorkerStartReason, number>>;
  stops_by_reason?: Partial<Record<WorkerStopReason, number>>;
  emptied_by_reason?: Partial<Record<WorkerStopReason, number>>;
  exits_without_stop?: number;
  last_start_reason?: WorkerStartReason | null;
  last_stop_reason?: WorkerStopReason | null;
  counted_since?: string | null;
}
