import { useEffect, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { WorkflowInstallProgress } from "./types";
import "./WorkflowInstallStatus.css";

const labels: Record<WorkflowInstallProgress["phase"], string> = {
  ready: "Downloads ready for review",
  downloading: "Downloading workflow files",
  paused: "Downloads paused",
  verifying: "Checking installed dependencies",
  needs_attention: "Installation needs attention",
  completed: "Installation completed",
  invalidated: "Installation changed",
  expired: "Installation offer expired",
};
const countFields = ["completed_downloads", "failed_downloads", "cancelled_downloads",
  "paused_downloads", "pending_downloads", "unavailable_downloads"] as const;

function validProgress(value: WorkflowInstallProgress, id: string, revision: string): boolean {
  return Boolean(value && value.id === id && value.workflow_revision_id === revision
    && Object.hasOwn(labels, value.phase)
    && ["ready", "queued", "invalidated", "completed", "expired"].includes(value.status)
    && (value.phase === "completed") === (value.status === "completed")
    && Number.isSafeInteger(value.total_downloads) && value.total_downloads >= 0
    && value.total_downloads <= 64
    && countFields.every(field => Number.isSafeInteger(value[field]) && value[field] >= 0)
    && countFields.reduce((sum, field) => sum + value[field], 0) <= value.total_downloads);
}

function heading(progress: WorkflowInstallProgress): string {
  return progress.attention_code === "workflow-extension-review-required" ? "Review extension code" : labels[progress.phase];
}

function guidance(progress: WorkflowInstallProgress): string {
  if (progress.phase === "completed" && progress.attention_code === "workflow-media-restore-failed") return "The workflow was installed, but the previous media setup could not be restored. Check worker status in Settings before generating.";
  if (progress.attention_code === "workflow-extension-review-required") return "Review the extension code in Extensions. Installation continues after approval.";
  if (progress.phase === "downloading") return "You can continue working while the files download.";
  if (progress.phase === "paused") return "Resume the paused downloads in Jobs.";
  if (progress.phase === "verifying") return "Downloads finished. The installed files are being checked.";
  if (progress.phase === "completed") return "Review workflow setup to check whether this workflow can run.";
  if (progress.phase === "expired" || progress.phase === "invalidated") return "Review workflow setup for a current installation offer.";
  if (progress.attention_code === "workflow-dependencies-need-selection"
      || progress.attention_code === "download-results-need-binding") {
    return "Choose the installed dependencies in workflow setup.";
  }
  if (progress.attention_code === "workflow-review-required") return "Review this workflow again before continuing setup.";
  if (progress.attention_code === "workflow-runtime-plan-changed") return "The media runtime changed. Review workflow setup before continuing.";
  if (progress.attention_code === "workflow-runtime-plan-unavailable") return "The media runtime is unavailable. Open workflow setup to check it.";
  if (progress.attention_code === "workflow-download-failed") return "A download stopped. Use Jobs to retry it.";
  return "Review workflow setup to resolve the installation problem.";
}

interface Props {
  progress?: WorkflowInstallProgress | null;
  workflowName: string;
  revisionId: string | null;
  onReviewSetup?: () => void;
  /** Another view of this installation announces it and offers its actions. */
  summary?: boolean;
}

export function WorkflowInstallStatus(props: Props) {
  const { progress, revisionId } = props;
  if (!progress || !revisionId || progress.workflow_revision_id !== revisionId) return null;
  return <BoundInstallStatus key={progress.id + ":" + revisionId}
    snapshot={progress} revisionId={revisionId} workflowName={props.workflowName}
    onReviewSetup={props.onReviewSetup} summary={props.summary} />;
}

function BoundInstallStatus({ snapshot, revisionId, workflowName, onReviewSetup, summary }: {
  snapshot: WorkflowInstallProgress; revisionId: string; workflowName: string; onReviewSetup?: () => void;
  summary?: boolean;
}) {
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["workflow-install-progress", snapshot.id],
    queryFn: async ({ signal }) => {
      const current = await api.workflowInstallProgress(snapshot.id, signal);
      if (!validProgress(current, snapshot.id, revisionId)) throw new Error("Invalid installation status");
      return current;
    },
    initialData: validProgress(snapshot, snapshot.id, revisionId) ? snapshot : undefined,
    retry: false,
    refetchInterval: query => !query.state.error && query.state.data?.status === "queued" ? 3_000 : false,
  });
  const previousPhase = useRef(snapshot.phase);
  useEffect(() => {
    const phase = query.data?.phase;
    if (phase === "completed" && previousPhase.current !== "completed") {
      for (const key of ["workflows", "workflow-families", "workflow-family", "studio-capabilities"]) {
        void client.invalidateQueries({ queryKey: [key] });
      }
    }
    if (phase) previousPhase.current = phase;
  }, [client, query.data?.phase]);
  const current = query.error ? undefined : query.data;
  if (summary) {
    // The same installation can show in the family list, its variants and the
    // workflow details at once. Only one of them is a live region with a
    // Refresh control, so a change is announced once and each control is unique.
    return (
      <div className="workflow-install-status">
        {query.error ? <p>Installation status is unavailable.</p> : current ? (
          <>
            <p><strong>{heading(current)}</strong></p>
            <p>{guidance(current)}</p>
            {current.total_downloads > 0
              && <small>{current.completed_downloads} of {current.total_downloads} downloads finished</small>}
          </>
        ) : <p>Checking installation status…</p>}
      </div>
    );
  }
  return (
    <section className="workflow-install-status" aria-label={"Installation for " + workflowName}>
      {query.error ? <p role="alert">Installation status is unavailable. Refresh to try again.</p> : current ? (
        <>
          <p role="status"><strong>{heading(current)}</strong></p>
          <p>{guidance(current)}</p>
          {current.total_downloads > 0 && <>
            <progress aria-label={"Downloads for " + workflowName}
              value={current.completed_downloads} max={current.total_downloads} />
            <small>{current.completed_downloads} of {current.total_downloads} downloads finished</small>
          </>}
        </>
      ) : <p role="status">Checking installation status…</p>}
      <div className="workflow-install-status-actions">
        <button className="secondary compact-button" aria-label="Refresh installation status"
          aria-disabled={query.isFetching} onClick={() => {
            if (!query.isFetching) void query.refetch();
          }}>Refresh</button>
        {onReviewSetup && <button className="secondary compact-button" onClick={onReviewSetup}>
          Review workflow setup
        </button>}
      </div>
    </section>
  );
}
