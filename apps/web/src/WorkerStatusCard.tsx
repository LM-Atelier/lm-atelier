import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "./api";
import { StatusDot } from "./StatusDot";
import { formatBytes } from "./format";
import type { WorkerStatus } from "./types";
import { workerFailureSummary } from "./workerFailures";

export function WorkerStatusCard({
  worker,
  startPending,
  stopPending,
  onStart,
  onStop,
}: {
  worker: WorkerStatus;
  startPending: boolean;
  stopPending: boolean;
  onStart: () => void;
  onStop: () => void;
}) {
  const client = useQueryClient();
  const [showLog, setShowLog] = useState(false);
  const refresh = () => {
    void client.invalidateQueries({ queryKey: ["workers"] });
    void client.invalidateQueries({ queryKey: ["jobs"] });
  };
  const restart = useMutation({
    mutationFn: () => api.restartWorker(worker.name),
    onSettled: refresh,
  });
  const reset = useMutation({
    mutationFn: () => api.resetWorker(worker.name),
    onSettled: refresh,
  });
  const logTail = useQuery({
    queryKey: ["worker-log-tail", worker.name],
    queryFn: () => api.workerLogTail(worker.name),
    enabled: showLog,
  });
  const busy = worker.active_jobs + worker.queued_jobs > 0;
  const busyTitle = busy ? "Wait for jobs to finish, or use Cancel jobs and reset" : undefined;
  const failed = worker.state === "exited";
  // Chat restarts with the model it ran last, so the record must know one;
  // media's stopped state is covered by the Start button instead.
  const restartable = worker.name === "chat" ? Boolean(worker.profile_id) : worker.running;
  const error = restart.error ?? reset.error;
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
          <button
            className="secondary compact-button"
            aria-label={`Show recent ${worker.name} worker log`}
            onClick={() => setShowLog((value) => !value)}
          >
            {showLog ? "Hide recent log" : "Show recent log"}
          </button>
          {showLog && logTail.data && (
            <pre className="worker-log-tail" aria-label={`${worker.name} worker recent log`}>
              {logTail.data.text || "The log is empty."}
            </pre>
          )}
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
            disabled={busy || restart.isPending}
            title={busyTitle}
            onClick={() => restart.mutate()}
          >
            {restart.isPending ? "Restarting…" : "Restart"}
          </button>
        )}
        {busy && (
          <button
            className="secondary danger compact-button"
            aria-label={`Cancel ${worker.name} jobs and reset the worker`}
            disabled={reset.isPending}
            title="Cancels this worker's queued and running jobs, then stops it"
            onClick={() => reset.mutate()}
          >
            {reset.isPending ? "Resetting…" : "Cancel jobs and reset"}
          </button>
        )}
      </div>
      {error && (
        <div className="callout error" role="alert">
          <span>{error.message}</span>
        </div>
      )}
    </article>
  );
}
