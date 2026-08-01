import { StatusDot } from "./StatusDot";
import { formatBytes } from "./format";
import type { WorkerStatus } from "./types";
import { workerFailureSummary } from "./workerFailures";

export function WorkerStatusCard({
  worker,
  startPending,
  stopPending,
  restartPending,
  resetPending,
  onStart,
  onStop,
  onRestart,
  onReset,
}: {
  worker: WorkerStatus;
  startPending: boolean;
  stopPending: boolean;
  restartPending: boolean;
  resetPending: boolean;
  onStart: () => void;
  onStop: () => void;
  onRestart: () => void;
  onReset: () => void;
}) {
  const busy = worker.active_jobs + worker.queued_jobs > 0;
  const busyTitle = busy ? "Wait for jobs to finish, or use Cancel jobs and reset" : undefined;
  const failed = worker.state === "exited";
  // Chat restarts with the model it ran last, so the record must know one;
  // media's stopped state is covered by the Start button instead.
  const restartable = worker.name === "chat" ? Boolean(worker.profile_id) : worker.running;
  return (
    <article className="engine-card">
      <header>
        <div>
          <h3>{worker.name} worker</h3>
          <p>
            {worker.state === "ready"
              ? `Ready · PID ${worker.pid}`
              : worker.state === "starting"
                ? "Starting and checking health"
                : failed
                  ? `Exited · code ${worker.exit_code ?? "unknown"}`
                  : "Stopped or externally managed"}
          </p>
        </div>
        <StatusDot healthy={worker.state === "ready"} />
      </header>
      <div className="worker-metrics">
        <span><strong>{worker.active_jobs}</strong> active</span>
        <span><strong>{worker.queued_jobs}</strong> queued</span>
        <span><strong>{formatBytes(worker.current_memory_bytes)}</strong> current RAM</span>
        <span><strong>{formatBytes(worker.peak_memory_bytes)}</strong> measured peak</span>
        {worker.estimated_memory_bytes != null && (
          <span><strong>{formatBytes(worker.estimated_memory_bytes)}</strong> estimated load</span>
        )}
      </div>
      {failed && (
        <div className="worker-failure" role="alert">
          <strong>{workerFailureSummary(worker)}</strong>
          {worker.failure_remedy && <p className="worker-remedy">{worker.failure_remedy}</p>}
          {worker.stderr_tail && (
            <details>
              <summary>What the engine reported</summary>
              <pre aria-label={`${worker.name} worker error output`}>{worker.stderr_tail}</pre>
            </details>
          )}
          {worker.log_path && <small>Log · Data folder/{worker.log_path}</small>}
        </div>
      )}
      <div className="capability-list">
        {worker.name === "media" && !worker.running && (
          <button
            className="secondary compact-button"
            aria-label={`Start ${worker.name} worker`}
            disabled={busy || startPending}
            title={busyTitle}
            onClick={onStart}
          >
            Start ComfyUI
          </button>
        )}
        {worker.running && (
          <button
            className="secondary compact-button"
            aria-label={`Unload ${worker.name} worker`}
            disabled={busy || stopPending}
            title={busyTitle}
            onClick={onStop}
          >
            Unload
          </button>
        )}
        {restartable && (
          <button
            className="secondary compact-button"
            aria-label={`Restart ${worker.name} worker`}
            disabled={busy || restartPending}
            title={busyTitle}
            onClick={onRestart}
          >
            {restartPending ? "Restarting…" : "Restart"}
          </button>
        )}
        {busy && (
          <button
            className="secondary danger compact-button"
            aria-label={`Cancel ${worker.name} jobs and reset the worker`}
            disabled={resetPending}
            title="Cancels this worker's queued and running jobs, then stops it"
            onClick={onReset}
          >
            {resetPending ? "Resetting…" : "Cancel jobs and reset"}
          </button>
        )}
      </div>
    </article>
  );
}
