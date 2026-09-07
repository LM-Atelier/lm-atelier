import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { api } from "./api";
import { WorkflowRevisionReviewPanel } from "./WorkflowRevisionReviewPanel";
import type { WorkflowRevisionReview } from "./types";

vi.mock("./api", () => ({
  api: {
    previewWorkflowRevisionReview: vi.fn(),
    decideWorkflowRevisionReview: vi.fn(),
  },
}));

const snapshot = (overrides: Partial<WorkflowRevisionReview> = {}): WorkflowRevisionReview => ({
  revision_id: "revision-a",
  subject_sha256: "a".repeat(64),
  trusted: false,
  can_approve: true,
  reasons: [],
  state: "unreviewed",
  reviewed_at: null,
  node_types: ["ExampleNode"],
  packages: [{ kind: "git", id: "example-nodes", source: "https://example.com/nodes", commit: "c".repeat(40), tree: "d".repeat(64) }],
  api_graph: { "1": { class_type: "ExampleNode", inputs: {} } },
  input_schema: { type: "object", properties: { steps: { type: "integer" } } },
  dependencies: { custom_nodes: ["example-nodes"] },
  ...overrides,
});

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const element = (workflowId: string, revisionId: string) => (
    <QueryClientProvider client={client}>
      <WorkflowRevisionReviewPanel key={`${workflowId}:${revisionId}`} workflowId={workflowId} revisionId={revisionId} />
    </QueryClientProvider>
  );
  const view = render(element("workflow-a", "revision-a"));
  return { client, select: (workflowId: string, revisionId: string) => view.rerender(element(workflowId, revisionId)) };
}

function openReview() {
  fireEvent.click(screen.getByText("Advanced"));
  fireEvent.click(screen.getByRole("button", { name: "Review exact revision" }));
}

const trustButton = () => screen.getByRole("button", { name: "Trust exact revision" });

describe("WorkflowRevisionReviewPanel", () => {
  beforeEach(() => {
    vi.mocked(api.previewWorkflowRevisionReview).mockResolvedValue(snapshot());
  });
  afterEach(() => {
    cleanup();
    vi.resetAllMocks();
  });

  it("fetches only after opening the review and never approves automatically", async () => {
    renderPanel();
    expect(api.previewWorkflowRevisionReview).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("Advanced"));
    expect(api.previewWorkflowRevisionReview).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Review exact revision" }));
    await waitFor(() => expect(trustButton()).toBeEnabled());
    expect(api.previewWorkflowRevisionReview).toHaveBeenCalledWith("workflow-a", "revision-a");
    expect(api.decideWorkflowRevisionReview).not.toHaveBeenCalled();
    expect(screen.getByText("https://example.com/nodes")).toBeInTheDocument();
    expect(screen.getByText("c".repeat(40))).toBeInTheDocument();
    expect(screen.getByText("d".repeat(64))).toBeInTheDocument();
    expect(screen.getByText(/"class_type": "ExampleNode"/)).toBeInTheDocument();
    expect(screen.getByText(/"steps"/)).toBeInTheDocument();
    expect(screen.getByText(/"custom_nodes"/)).toBeInTheDocument();
  });

  it("sends only the exact snapshot digest and explicit action, then refreshes consumers", async () => {
    const approved = snapshot({ trusted: true, can_approve: true, state: "reviewed" });
    vi.mocked(api.decideWorkflowRevisionReview).mockResolvedValue(approved);
    const { client } = renderPanel();
    const invalidate = vi.spyOn(client, "invalidateQueries");
    openReview();
    await waitFor(() => expect(trustButton()).toBeEnabled());
    vi.mocked(api.previewWorkflowRevisionReview).mockResolvedValue(approved);
    fireEvent.click(trustButton());
    await waitFor(() => expect(api.decideWorkflowRevisionReview).toHaveBeenCalledExactlyOnceWith(
      "workflow-a", "revision-a", { action: "approve", subject_sha256: "a".repeat(64) },
    ));
    await waitFor(() => expect(screen.getByRole("button", { name: "Revoke review" })).toBeEnabled());
    for (const key of ["workflows", "workflow-families", "workflow-family", "studio-capabilities"]) {
      expect(invalidate).toHaveBeenCalledWith({ queryKey: [key] });
    }
    expect(api.previewWorkflowRevisionReview).toHaveBeenCalledTimes(2);
    expect(trustButton()).toBeDisabled();
  });

  it("revokes an existing review with the reviewed snapshot identity", async () => {
    vi.mocked(api.previewWorkflowRevisionReview).mockResolvedValue(snapshot({ trusted: true }));
    vi.mocked(api.decideWorkflowRevisionReview).mockResolvedValue(snapshot());
    renderPanel();
    openReview();
    const revoke = screen.getByRole("button", { name: "Revoke review" });
    await waitFor(() => expect(revoke).toBeEnabled());
    fireEvent.click(revoke);
    await waitFor(() => expect(api.decideWorkflowRevisionReview).toHaveBeenCalledExactlyOnceWith(
      "workflow-a", "revision-a", { action: "revoke", subject_sha256: "a".repeat(64) },
    ));
  });

  it("allows revoking an approved review when runtime unavailability makes it untrusted", async () => {
    vi.mocked(api.previewWorkflowRevisionReview).mockResolvedValue(snapshot({
      trusted: false,
      state: "approved",
      can_approve: false,
      reasons: ["workflow_review_runtime_unavailable"],
    }));
    vi.mocked(api.decideWorkflowRevisionReview).mockResolvedValue(snapshot({ state: "revoked" }));
    renderPanel();
    openReview();
    await screen.findByText(/Start the media runtime/);
    expect(trustButton()).toBeDisabled();
    const revoke = screen.getByRole("button", { name: "Revoke review" });
    await waitFor(() => expect(revoke).toBeEnabled());
    fireEvent.click(revoke);
    await waitFor(() => expect(api.decideWorkflowRevisionReview).toHaveBeenCalledExactlyOnceWith(
      "workflow-a", "revision-a", { action: "revoke", subject_sha256: "a".repeat(64) },
    ));
  });

  it.each([
    ["workflow_review_runtime_unavailable", /Start the media runtime/],
    ["workflow_review_node_unavailable", /Install and review their packages/],
  ])("refuses approval when %s and explains a remedy", async (reason, remedy) => {
    vi.mocked(api.previewWorkflowRevisionReview).mockResolvedValue(snapshot({ can_approve: false, reasons: [String(reason)] }));
    renderPanel();
    openReview();
    await screen.findByText(remedy);
    expect(trustButton()).toBeDisabled();
    fireEvent.click(trustButton());
    expect(api.decideWorkflowRevisionReview).not.toHaveBeenCalled();
  });

  it("keeps decisions disabled when a preview fails", async () => {
    vi.mocked(api.previewWorkflowRevisionReview).mockRejectedValue(new Error("Review request failed"));
    renderPanel();
    openReview();
    expect(trustButton()).toBeDisabled();
    await screen.findByText("Review request failed");
    expect(trustButton()).toBeDisabled();
    expect(screen.getByRole("button", { name: "Revoke review" })).toBeDisabled();
    expect(api.decideWorkflowRevisionReview).not.toHaveBeenCalled();
  });

  it("disables the previous snapshot while refreshing and after a failed refresh", async () => {
    renderPanel();
    openReview();
    await waitFor(() => expect(trustButton()).toBeEnabled());
    let rejectPreview!: (error: Error) => void;
    vi.mocked(api.previewWorkflowRevisionReview).mockImplementationOnce(() => new Promise((_resolve, reject) => { rejectPreview = reject; }));
    fireEvent.click(screen.getByRole("button", { name: "Refresh review" }));
    await waitFor(() => expect(trustButton()).toBeDisabled());
    rejectPreview(new Error("Runtime connection lost"));
    await screen.findByText("Runtime connection lost");
    expect(trustButton()).toBeDisabled();
    fireEvent.click(trustButton());
    expect(api.decideWorkflowRevisionReview).not.toHaveBeenCalled();
  });

  it("requires a new preview after a changed-snapshot refusal", async () => {
    vi.mocked(api.decideWorkflowRevisionReview).mockRejectedValue(Object.assign(new Error("Conflict"), { code: "workflow-review-changed" }));
    renderPanel();
    openReview();
    await waitFor(() => expect(trustButton()).toBeEnabled());
    fireEvent.click(trustButton());
    await screen.findByText(/Refresh the review and inspect the new snapshot/);
    expect(trustButton()).toBeDisabled();
    vi.mocked(api.previewWorkflowRevisionReview).mockResolvedValue(snapshot({ subject_sha256: "b".repeat(64) }));
    fireEvent.click(screen.getByRole("button", { name: "Refresh review" }));
    await waitFor(() => expect(trustButton()).toBeEnabled());
    fireEvent.click(trustButton());
    await waitFor(() => expect(api.decideWorkflowRevisionReview).toHaveBeenLastCalledWith(
      "workflow-a", "revision-a", { action: "approve", subject_sha256: "b".repeat(64) },
    ));
  });

  it.each([
    { revision_id: "another-revision" },
    { subject_sha256: "" },
  ])("refuses an incomplete or mismatched snapshot: %j", async (overrides) => {
    vi.mocked(api.previewWorkflowRevisionReview).mockResolvedValue(snapshot(overrides));
    renderPanel();
    openReview();
    await screen.findByText("Required node types");
    expect(trustButton()).toBeDisabled();
    expect(api.decideWorkflowRevisionReview).not.toHaveBeenCalled();
  });

  it.each([
    ["workflow-a", "revision-b"],
    ["workflow-b", "revision-a"],
  ])("resets review on selection of %s/%s and ignores the previous selection", async (workflowId, revisionId) => {
    const { select } = renderPanel();
    openReview();
    await waitFor(() => expect(trustButton()).toBeEnabled());
    select(workflowId, revisionId);
    expect(screen.queryByRole("button", { name: "Trust exact revision" })).toBeNull();
    expect(api.previewWorkflowRevisionReview).toHaveBeenCalledTimes(1);
    vi.mocked(api.previewWorkflowRevisionReview).mockResolvedValue(snapshot({ revision_id: revisionId, subject_sha256: "b".repeat(64) }));
    openReview();
    await waitFor(() => expect(trustButton()).toBeEnabled());
    fireEvent.click(trustButton());
    await waitFor(() => expect(api.decideWorkflowRevisionReview).toHaveBeenCalledWith(
      workflowId, revisionId, { action: "approve", subject_sha256: "b".repeat(64) },
    ));
  });
});
