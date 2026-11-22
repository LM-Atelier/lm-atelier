import { useEffect } from "react";
import { api } from "./api";
import { EDITOR_HEARTBEAT_MS, type WorkflowEditorSubmission } from "./workflowEditorBridge";

/** The HTTP status an error carries, read from its shape so a stand-in error counts too. */
function answeredStatus(error: unknown): number | null {
  const status = typeof error === "object" && error !== null
    ? (error as { status?: unknown }).status
    : undefined;
  return typeof status === "number" ? status : null;
}

/** Whether a retry could still validate a returned edit whose latest attempt failed with `failure`.
 *
 * Nothing has failed yet, the request got no answer, or the server could not
 * do its part, as when the media runtime cannot describe its nodes: each can
 * pass on a later try. A refusal below 500 is about the edit or its session,
 * and a retry would meet it again. When the editor run keeps the ComfyUI
 * window open it passes the refusal on wrapped, as the wrapper's cause.
 */
function retryCouldSucceed(failure: unknown): boolean {
  if (failure === null || failure === undefined) return true;
  const cause = typeof failure === "object" ? (failure as { cause?: unknown }).cause : undefined;
  const status = answeredStatus(failure) ?? answeredStatus(cause);
  return status === null || status >= 500;
}

/** Report a native editor session while this page holds its returned edit and a retry could validate it.
 *
 * The server counts an open editor session as use of the media worker only
 * while the page holding it reports in, and the editor run stops reporting
 * when it ends. A returned edit whose validation failed stays on this page,
 * and its retry is refused on any other media launch, so the session goes on
 * being reported until the retry succeeds, the edit is discarded or the view
 * holding it unmounts. The reports pause while `lastFailure`, the latest
 * reason the edit is held, is a refusal a retry would meet again: holding the
 * launch for an edit that can never be saved would only keep media runs
 * waiting. A later retry that fails in a way another could fix resumes them.
 */
export function useWorkflowEditorReports(
  submission: WorkflowEditorSubmission | null,
  lastFailure: unknown,
): void {
  const workflowId = submission?.workflowId;
  const sessionId = submission?.sessionId;
  const nonce = submission?.nonce;
  const retryable = retryCouldSucceed(lastFailure);
  useEffect(() => {
    if (!retryable || workflowId === undefined || sessionId === undefined || nonce === undefined) return;
    const timer = window.setInterval(() => {
      // Nothing here depends on the answer. Once the session is consumed,
      // cancelled or expired the server says it is gone, and it has stopped
      // counting as use either way.
      void api.heartbeatWorkflowEditor(workflowId, sessionId, nonce).catch(() => undefined);
    }, EDITOR_HEARTBEAT_MS);
    return () => window.clearInterval(timer);
  }, [workflowId, sessionId, nonce, retryable]);
}
