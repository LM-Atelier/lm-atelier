import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";

const REASONS: Record<string, string> = {
  workflow_review_runtime_unavailable:
    "Start the media runtime, then refresh this review to check node availability.",
  workflow_review_node_unavailable:
    "Required node types are unavailable. Install and review their packages, activate the required nodes, then refresh this review.",
};

function reviewError(error: Error): string {
  const code = (error as Error & { code?: string }).code;
  if (code === "workflow-review-changed") {
    return "The revision or its packages changed. Refresh the review and inspect the new snapshot before trusting it.";
  }
  if (code === "workflow-review-unavailable") {
    return "The revision cannot be reviewed right now. Check the media runtime and required nodes, then refresh this review.";
  }
  return error.message;
}

function identity(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value) ?? "Not recorded";
}

function RevisionReview({ workflowId, revisionId }: { workflowId: string; revisionId: string }) {
  const client = useQueryClient();
  const queryKey = ["workflow-revision-review", workflowId, revisionId];
  const preview = useQuery({
    queryKey,
    queryFn: () => api.previewWorkflowRevisionReview(workflowId, revisionId),
    retry: false,
    gcTime: 0,
  });
  const snapshot = preview.data;
  // Cached data and failed refreshes are display evidence only. Every decision
  // must refer to the successful preview fetched for this mounted selection.
  const snapshotReady = preview.isSuccess && preview.isFetchedAfterMount
    && !preview.isFetching && snapshot?.revision_id === revisionId
    && /^[a-f0-9]{64}$/i.test(snapshot.subject_sha256);
  const decision = useMutation({
    mutationFn: (action: "approve" | "revoke") => {
      if (!snapshotReady || !snapshot || (action === "approve" && !snapshot.can_approve)) {
        throw new Error("Refresh the review before making a decision.");
      }
      return api.decideWorkflowRevisionReview(workflowId, revisionId, {
        action,
        subject_sha256: snapshot.subject_sha256,
      });
    },
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: ["workflows"] }),
        client.invalidateQueries({ queryKey: ["workflow-families"] }),
        client.invalidateQueries({ queryKey: ["workflow-family"] }),
        client.invalidateQueries({ queryKey: ["studio-capabilities"] }),
        client.invalidateQueries({ queryKey }),
      ]);
    },
  });
  const canDecide = snapshotReady && !decision.isPending && !decision.isError;
  return (
    <section className="workflow-input-section" aria-label="Exact revision review">
      <p className="package-review-note">
        Review this executable graph, input schema, dependencies, and package identities.
        Trust applies to this exact snapshot. Changes require another review.
      </p>
      {preview.isFetching && <p role="status">Loading exact revision review…</p>}
      {preview.error && <ErrorCallout message={reviewError(preview.error)} />}
      {decision.error && <ErrorCallout message={reviewError(decision.error)} />}
      {snapshot && (
        <>
          <dl className="confirm-facts">
            <div><dt>Revision</dt><dd><code>{snapshot.revision_id}</code></dd></div>
            <div><dt>Snapshot SHA-256</dt><dd><code>{snapshot.subject_sha256}</code></dd></div>
            <div><dt>Review state</dt><dd>{snapshot.state}</dd></div>
            <div><dt>Reviewed at</dt><dd>{snapshot.reviewed_at ?? "Not reviewed"}</dd></div>
          </dl>
          {snapshot.reasons.length > 0 && (
            <ul className="package-review-note">
              {snapshot.reasons.map((reason) => <li key={reason}>{REASONS[reason] ?? reason.replaceAll("_", " ")}</li>)}
            </ul>
          )}
          <h4>Required node types</h4>
          <p>{snapshot.node_types.join(", ") || "None declared"}</p>
          <h4>Package identities</h4>
          {snapshot.packages.length === 0 && <p>No verified custom packages in this snapshot.</p>}
          {snapshot.packages.map((pkg, index) => (
            <dl className="confirm-facts" key={`${identity(pkg.id)}-${index}`}>
              <div><dt>Package</dt><dd>{identity(pkg.package_id ?? pkg.id)}</dd></div>
              <div><dt>Source</dt><dd>{identity(pkg.source)}</dd></div>
              {pkg.kind === "git" ? (
                <>
                  <div><dt>Commit</dt><dd><code>{identity(pkg.commit)}</code></dd></div>
                  <div><dt>Tree</dt><dd><code>{identity(pkg.tree)}</code></dd></div>
                </>
              ) : (
                <>
                  <div><dt>Version</dt><dd>{identity(pkg.version)}</dd></div>
                  <div><dt>Archive</dt><dd><code>{identity(pkg.archive)}</code></dd></div>
                  <div><dt>Manifest</dt><dd><code>{identity(pkg.manifest)}</code></dd></div>
                </>
              )}
            </dl>
          ))}
          <details open><summary>Review snapshot: executable graph</summary><pre>{JSON.stringify(snapshot.api_graph, null, 2)}</pre></details>
          <details><summary>Review snapshot: input schema</summary><pre>{JSON.stringify(snapshot.input_schema, null, 2)}</pre></details>
          <details><summary>Review snapshot: dependencies</summary><pre>{JSON.stringify(snapshot.dependencies, null, 2)}</pre></details>
        </>
      )}
      <div className="row-actions">
        <button
          type="button"
          className="secondary compact-button"
          disabled={preview.isFetching || decision.isPending}
          onClick={() => { decision.reset(); void preview.refetch(); }}
        >Refresh review</button>
        <button
          type="button"
          className="primary compact-button"
          disabled={!canDecide || !snapshot?.can_approve || snapshot.trusted}
          onClick={() => decision.mutate("approve")}
        >Trust exact revision</button>
        <button
          type="button"
          className="secondary compact-button danger"
          disabled={!canDecide || (!snapshot?.trusted && snapshot?.state !== "approved")}
          onClick={() => decision.mutate("revoke")}
        >Revoke review</button>
      </div>
    </section>
  );
}

export function WorkflowRevisionReviewPanel({ workflowId, revisionId }: { workflowId: string; revisionId: string }) {
  const [open, setOpen] = useState(false);
  return (
    <details>
      <summary>Advanced</summary>
      <button
        type="button"
        className="secondary compact-button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >Review exact revision</button>
      {open && <RevisionReview key={`${workflowId}:${revisionId}`} workflowId={workflowId} revisionId={revisionId} />}
    </details>
  );
}
