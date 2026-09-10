import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, CircleStop, Pause, Play, RotateCcw } from "lucide-react";
import { api } from "./api";
import { QueueActivityDialog } from "./QueueActivityDialog";
import { jobProgressFraction, jobProgressText } from "./jobProgress";
import "./JobsPanel.css";

const RECENT_UNSUCCESSFUL_JOB_LIMIT = 3;
const DISMISSED_JOB_ISSUES_KEY = "lm-atelier-dismissed-job-issues-before";

function jobDisplayName(kind: string): string {
  if (kind === "edit_verify") return "Image edit check";
  if (kind === "registry_prepare") return "Package preparation";
  return kind;
}

export function JobsPanel() {
  const client = useQueryClient();
  const [queueOpen, setQueueOpen] = useState(false);
  const queueButton = useRef<HTMLButtonElement>(null);
  const [dismissedBefore, setDismissedBefore] = useState(() => {
    const saved = Number(localStorage.getItem(DISMISSED_JOB_ISSUES_KEY));
    return Number.isFinite(saved) && saved > 0 ? saved : 0;
  });
  const [activeLimit, setActiveLimit] = useState(100);
  const jobs = useQuery({
    queryKey: ["jobs", "activity", activeLimit], queryFn: () => api.jobActivity(activeLimit),
    placeholderData: keepPreviousData, refetchInterval: 3_000,
  });
  const refresh = () => void client.invalidateQueries({ queryKey: ["jobs"] });
  const cancel = useMutation({ mutationFn: api.cancelJob, onSuccess: refresh });
  const pause = useMutation({ mutationFn: api.pauseDownload, onSuccess: refresh });
  const resume = useMutation({ mutationFn: api.resumeDownload, onSuccess: refresh });
  const retry = useMutation({ mutationFn: api.retryJob, onSuccess: refresh });
  const visibleJobs = jobs.data?.active.filter((job) => job.kind !== "edit_verify") ?? [];
  const active = visibleJobs.filter((job) => ["queued", "running", "paused"].includes(job.status));
  const activeCount = jobs.data?.active_count ?? 0;
  const recentUnsuccessful = (jobs.data?.recent_issues ?? [])
    .filter((job) => job.kind !== "edit_verify")
    .filter((job) => ["failed", "cancelled", "interrupted"].includes(job.status))
    .filter((job) => Date.parse(job.updated_at) > dismissedBefore)
    .sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at))
    .slice(0, RECENT_UNSUCCESSFUL_JOB_LIMIT);
  const clearRecentIssues = () => {
    const newestIssue = Math.max(
      dismissedBefore,
      ...recentUnsuccessful
        .map((job) => Date.parse(job.updated_at))
        .filter((updatedAt) => Number.isFinite(updatedAt)),
    );
    const cutoff = newestIssue;
    localStorage.setItem(DISMISSED_JOB_ISSUES_KEY, String(cutoff));
    setDismissedBefore(cutoff);
  };
  const queueEntry = (
    <button ref={queueButton} className={"secondary compact-button queue-activity-entry"
      + (!jobs.error && !activeCount && !recentUnsuccessful.length ? " queue-activity-entry-idle" : "")}
      onClick={() => setQueueOpen(true)}>View accepted work</button>
  );
  const closeQueue = () => {
    setQueueOpen(false);
    queueButton.current?.focus();
  };
  const queueDialog = queueOpen
    ? createPortal(<QueueActivityDialog onClose={closeQueue} />, document.body, "accepted-work")
    : null;
  // A jobs read that failed is not "no jobs". Returning null took away the
  // one surface that would have explained what was happening, at exactly the
  // moment something was - and left the workspace looking idle instead.
  if (jobs.error) {
    return (
      <>
      <aside className="jobs-panel" aria-label="Jobs">
        <div className="jobs-error" role="alert">
          {(jobs.error as Error).message}
          <button className="secondary compact-button" onClick={() => void jobs.refetch()}>
            Try again
          </button>
        </div>
        {queueEntry}
      </aside>
      {queueDialog}
      </>
    );
  }
  if (!activeCount && !recentUnsuccessful.length) return <>{queueEntry}{queueDialog}</>;
  return (
    <>
    <aside className="jobs-panel" aria-label="Jobs">
      <header>
        <Activity size={16} />
        <span>
          {activeCount
            ? `${activeCount} active job${activeCount === 1 ? "" : "s"}`
            : "Recent job issues"}
        </span>
        {recentUnsuccessful.length > 0 && (
          <button
            className="jobs-clear"
            aria-label="Clear recent job issues"
            onClick={clearRecentIssues}
          >
            Clear
          </button>
        )}
      </header>
      {active.length < activeCount && (
        <div className="jobs-page-summary">
          <span>Showing {active.length} of {activeCount} active jobs.</span>
          {activeLimit < 500 && (
            <button className="secondary compact-button" disabled={jobs.isFetching}
              onClick={() => setActiveLimit((limit) => Math.min(500, limit + 100))}>
              Show more active jobs
            </button>
          )}
        </div>
      )}
      {active.map((job) => (
        <div className="job-row" key={job.id}>
          <div>
            <strong>{jobDisplayName(job.kind)}</strong>
            <small>{jobProgressText(job)}</small>
            <div
              className="progress-track"
              role="progressbar"
              aria-label={`${jobDisplayName(job.kind)} progress`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={jobProgressFraction(job) === null
                ? undefined
                : Math.round(jobProgressFraction(job)! * 100)}
            >
              <div
                className={jobProgressFraction(job) === null ? "indeterminate" : undefined}
                style={jobProgressFraction(job) === null
                  ? undefined
                  : { width: `${jobProgressFraction(job)! * 100}%` }}
              />
            </div>
          </div>
          <span className="job-actions">
            {job.kind === "download" && (job.status === "paused" ? (
              <button
                className="icon-button"
                aria-label="Resume download"
                disabled={resume.isPending && resume.variables === job.id}
                onClick={() => resume.mutate(job.id)}
              >
                <Play size={16} />
              </button>
            ) : (
              <button
                className="icon-button"
                aria-label="Pause download"
                disabled={pause.isPending && pause.variables === job.id}
                onClick={() => pause.mutate(job.id)}
              >
                <Pause size={16} />
              </button>
            ))}
            {job.cancellable && (
              <button
                className="icon-button"
                aria-label="Cancel job"
                disabled={cancel.isPending && cancel.variables === job.id}
                onClick={() => cancel.mutate(job.id)}
              >
                <CircleStop size={17} />
              </button>
            )}
          </span>
        </div>
      ))}
      {active.length > 0 && recentUnsuccessful.length > 0 && (
        <div className="jobs-subheading">Recent issues</div>
      )}
      {recentUnsuccessful.map((job) => (
        <div className="job-row unsuccessful" key={job.id}>
          <div>
            <strong>{jobDisplayName(job.kind)} · {job.status}</strong>
            <small>{job.phase || "Stopped"}</small>
            {job.error && <small className="job-error" title={job.error}>{job.error}</small>}
          </div>
          <span className="job-actions">
            <button
              className="icon-button"
              aria-label={`Retry ${job.kind} job`}
              disabled={retry.isPending && retry.variables === job.id}
              onClick={() => retry.mutate(job.id)}
            >
              <RotateCcw size={16} />
            </button>
          </span>
        </div>
      ))}
      {(retry.error || cancel.error || pause.error || resume.error) && (
        <div className="jobs-error" role="alert">
          {(retry.error ?? cancel.error ?? pause.error ?? resume.error)!.message}
        </div>
      )}
      {queueEntry}
    </aside>
    {queueDialog}
    </>
  );
}
