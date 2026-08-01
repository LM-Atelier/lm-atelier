import type { WorkerStatus } from "./types";

/**
 * A headline for a failed worker that says what happened, not what exited.
 *
 * "chat worker exited with code 1" is true and tells nobody anything. When the
 * backend recognised the failure, lead with the recognisable part; the exit
 * detail and the engine output are still shown underneath.
 */
const HEADLINES: Record<string, (name: string) => string> = {
  oom_vram: (name) => `The ${name} model needs more graphics memory than this computer has free.`,
  oom_host: (name) => `The ${name} model needs more system memory than this computer has free.`,
  port_in_use: (name) => `The ${name} worker could not start because its port is already in use.`,
  model_incompatible: (name) => `The ${name} engine could not read the selected model.`,
  executable_missing: (name) => `The ${name} engine program could not be started.`,
  startup_timeout: (name) => `The ${name} worker took too long to start.`,
  crashed: (name) => `The ${name} worker stopped unexpectedly.`,
};

export function workerFailureSummary(worker: WorkerStatus): string {
  const headline = worker.failure_code ? HEADLINES[worker.failure_code] : undefined;
  if (headline) return headline(worker.name);
  return worker.failure_detail || `${worker.name} worker stopped unexpectedly.`;
}
