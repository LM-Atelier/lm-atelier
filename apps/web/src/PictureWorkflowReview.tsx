import { useState } from "react";
import { createPortal } from "react-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import type { PackageReviewState } from "./useWorkflowPackageImport";
import { WorkflowPackageReview } from "./WorkflowPackageReview";

/** Wording for a picture whose workflow the review cannot be given, by the server's code. */
const FAILURES: Record<string, string> = {
  "picture-workflow-missing": "This picture's file carries no workflow that can be reviewed.",
  "generation-settings-unreadable": "The settings text stored in this picture could not be read.",
};

function editorGraph(value: unknown): Record<string, unknown> {
  const graph = value && typeof value === "object"
    ? (value as { ui_graph?: unknown }).ui_graph
    : undefined;
  if (!graph || typeof graph !== "object" || !Array.isArray((graph as { nodes?: unknown }).nodes)) {
    throw new Error(FAILURES["picture-workflow-missing"]);
  }
  return graph as Record<string, unknown>;
}

/**
 * Hand the workflow a picture's file carries to the review a workflow file gets.
 *
 * Nothing is imported until the person confirms it there, and what is imported
 * is untrusted: it runs only once it is reviewed in Workflows.
 */
export function PictureWorkflowReview({
  artifactId,
  pictureName,
}: {
  artifactId: string;
  pictureName?: string;
}) {
  const client = useQueryClient();
  const [review, setReview] = useState<PackageReviewState | null>(null);
  const [imported, setImported] = useState(false);
  const open = useMutation({
    mutationFn: async (): Promise<PackageReviewState> => {
      const uiGraph = editorGraph(await api.pictureWorkflow(artifactId));
      return {
        analysis: await api.analyzeWorkflowPackage(uiGraph),
        fileName: pictureName ? `${pictureName} workflow` : "Picture workflow",
        uiGraph,
      };
    },
    onSuccess: (state) => {
      setImported(false);
      setReview(state);
    },
  });
  const failure = open.error as (Error & { code?: string }) | null;
  return <>
    <button
      type="button"
      className="secondary compact-button"
      aria-disabled={open.isPending || undefined}
      onClick={() => {
        if (!open.isPending) open.mutate();
      }}
    >
      Review this picture's workflow
    </button>
    {failure && <ErrorCallout message={FAILURES[failure.code ?? ""] ?? failure.message} />}
    {imported && <p role="status">Added to Workflows. It runs only after you review it there.</p>}
    {/* Outside the card it is opened from, so the card's own styles never reach the dialog. */}
    {review && createPortal(
      <WorkflowPackageReview
        analysis={review.analysis}
        fileName={review.fileName}
        uiGraph={review.uiGraph}
        onImported={() => {
          setReview(null);
          setImported(true);
          void client.invalidateQueries({ queryKey: ["workflows"] });
        }}
        onClose={() => setReview(null)}
      />,
      document.body,
    )}
  </>;
}
