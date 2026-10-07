export type InstallQueueAction = "pause_after_current" | "resume";

export interface InstallQueuePolicy {
  lane: "install";
  dispatch_state: "open" | "draining" | "paused";
  revision: number;
  running_jobs: number;
  allowed_actions: InstallQueueAction[];
}
