import { mockWorkflowReadsFromFixture } from "./workflowReadFixtures";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import type { Workflow, WorkflowFamily, WorkflowInstallProgress } from "./types";

vi.mock("./api", () => ({ api: {
  workflows: vi.fn(), workflowSummaries: vi.fn(), workflow: vi.fn(), workflowFamilies: vi.fn(),
  workflowInstallProgress: vi.fn(), installWorkflowOffer: vi.fn(),
} }));
vi.mock("./CustomNodesPanel", () => ({ CustomNodesPanel: () => null }));
vi.mock("./RegistryInstallsPanel", () => ({ RegistryInstallsPanel: () => null }));
vi.mock("./WorkflowRevisionReviewPanel", () => ({
  WorkflowRevisionReviewPanel: () => <section aria-label="Workflow review">Review this workflow</section>,
}));
vi.mock("./WorkflowActivationPanel", () => ({
  WorkflowActivationPanel: () => <section aria-label="Workflow dependencies">Choose installed dependencies</section>,
}));

const stamp = "2026-09-10T00:00:00Z";
const workflow: Workflow = {
  id: "detail", family_id: "collection", name: "Detail variant", description: "Neutral fixture",
  operation: "text_to_image", current_revision_id: "revision-a",
  revisions: [{ id: "revision-a", workflow_id: "detail", version: 1, engine: "comfyui",
    engine_version: null, ui_graph_json: {}, api_graph_json: {}, input_schema_json: {},
    dependencies_json: {}, dependency_contract_sha256: "b".repeat(64), trusted: true, created_at: stamp }],
};
function progress(phase: WorkflowInstallProgress["phase"] = "downloading"): WorkflowInstallProgress {
  return { id: "offer-internal", workflow_revision_id: "revision-a", status: phase === "completed" ? "completed" : "queued",
    phase, total_downloads: 2, completed_downloads: phase === "completed" || phase === "verifying" ? 2 : 1,
    failed_downloads: 0, cancelled_downloads: 0, paused_downloads: phase === "paused" ? 1 : 0,
    pending_downloads: phase === "downloading" ? 1 : 0, unavailable_downloads: 0,
    attention_code: phase === "needs_attention" ? "workflow-dependencies-need-selection" : null };
}
function family(snapshot: WorkflowInstallProgress): WorkflowFamily {
  return { id: "collection", name: "Detail collection", description: "", use_case: "", tags: [],
    enabled: true, archived: false, compatibility: false, created_at: stamp, updated_at: stamp, preferences: [],
    variants: [{ id: workflow.id, variant_key: "image", name: workflow.name, operation: "text_to_image",
      current_revision_id: "revision-a", current_revision_version: 1, engine: "comfyui", capabilities: ["image"],
      trusted: true, readiness: "setup_required", readiness_reason: "activation_not_ready",
      setup_resolution: "attention_required", install_offer: null, install_progress: snapshot }] };
}
const clients: QueryClient[] = [];
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  return render(<QueryClientProvider client={client}><WorkflowsView /></QueryClientProvider>);
}
function arrange(snapshot: WorkflowInstallProgress) {
  vi.mocked(api.workflowFamilies).mockResolvedValue([family(snapshot)]);
  vi.mocked(api.workflowInstallProgress).mockResolvedValue(snapshot);
}
beforeEach(() => {
  vi.resetAllMocks();
  mockWorkflowReadsFromFixture(() => api.workflows());
  vi.mocked(api.workflows).mockResolvedValue([workflow]);
  arrange(progress());
});
afterEach(() => { cleanup(); clients.splice(0).forEach(client => client.clear()); });

it.each([
  ["downloading", "Downloading workflow files"],
  ["paused", "Downloads paused"],
  ["verifying", "Checking installed dependencies"],
  ["needs_attention", "Installation needs attention"],
  ["completed", "Installation completed"],
] as const)("shows durable %s progress after loading the Workflows page", async (phase, label) => {
  arrange(progress(phase));
  show();
  await screen.findByRole("heading", { name: "Detail collection" });
  const panel = screen.getByRole("region", { name: "Installation for Detail variant" });
  expect(within(panel).getByRole("status")).toHaveTextContent(label);
  expect(within(panel).getByText(/of 2 downloads finished/)).toBeInTheDocument();
  expect(panel).not.toHaveTextContent("offer-internal");
  expect(panel).not.toHaveTextContent("workflow-dependencies-need-selection");
  expect(panel).not.toHaveTextContent("Ready to run");
  expect(api.installWorkflowOffer).not.toHaveBeenCalled();
});

it("keeps an installed workflow completed while explaining a worker restoration failure", async () => {
  arrange({ ...progress("completed"), attention_code: "workflow-media-restore-failed" });
  show();
  await screen.findByRole("heading", { name: "Detail collection" });
  const panel = screen.getByRole("region", { name: "Installation for Detail variant" });
  expect(within(panel).getByRole("status")).toHaveTextContent("Installation completed");
  expect(panel).toHaveTextContent("The workflow was installed, but the previous media setup could not be restored.");
  expect(panel).toHaveTextContent("Check worker status in Settings before generating.");
  expect(panel).not.toHaveTextContent("workflow-media-restore-failed");
  expect(api.installWorkflowOffer).not.toHaveBeenCalled();
});

it("reloads persisted progress without relying on an earlier completion event", async () => {
  arrange(progress("needs_attention"));
  const first = show();
  await screen.findByRole("heading", { name: "Detail collection" });
  expect(screen.getByRole("region", { name: "Installation for Detail variant" })).toHaveTextContent("Installation needs attention");
  first.unmount();
  show();
  await screen.findByRole("heading", { name: "Detail collection" });
  expect(screen.getByRole("region", { name: "Installation for Detail variant" })).toHaveTextContent("Choose the installed dependencies");
});

it("opens the workflow setup from its attention message without queuing another download", async () => {
  arrange(progress("needs_attention"));
  show();
  await screen.findByRole("heading", { name: "Detail collection" });
  const panel = screen.getByRole("region", { name: "Installation for Detail variant" });
  fireEvent.click(within(panel).getByRole("button", { name: "Review workflow setup" }));
  expect(await screen.findByRole("region", { name: "Workflow dependencies" })).toBeInTheDocument();
  expect(api.installWorkflowOffer).not.toHaveBeenCalled();
});

it("refreshes an installation from the server without making an install request", async () => {
  show();
  await screen.findByRole("heading", { name: "Detail collection" });
  const panel = screen.getByRole("region", { name: "Installation for Detail variant" });
  await waitFor(() => expect(api.workflowInstallProgress).toHaveBeenCalled());
  vi.mocked(api.workflowInstallProgress).mockResolvedValue(progress("completed"));
  fireEvent.click(within(panel).getByRole("button", { name: "Refresh installation status" }));
  await waitFor(() => expect(within(panel).getByRole("status")).toHaveTextContent("Installation completed"));
  expect(api.installWorkflowOffer).not.toHaveBeenCalled();
});

it.each(["mismatched", "unavailable"])("does not claim completion after a %s status response", async (scenario) => {
  arrange(progress("completed"));
  if (scenario === "mismatched") vi.mocked(api.workflowInstallProgress).mockResolvedValue({
    ...progress("completed"), workflow_revision_id: "different-revision",
  });
  else vi.mocked(api.workflowInstallProgress).mockRejectedValue(new Error("neutral-server-detail-marker"));
  show();
  await screen.findByRole("heading", { name: "Detail collection" });
  const panel = screen.getByRole("region", { name: "Installation for Detail variant" });
  expect(await within(panel).findByRole("alert")).toHaveTextContent("Installation status is unavailable.");
  expect(panel).not.toHaveTextContent("Installation completed");
  expect(panel).not.toHaveTextContent("neutral-server-detail-marker");
});

it("uses ordinary guidance for an unknown attention code", async () => {
  arrange(JSON.parse(JSON.stringify({ ...progress("needs_attention"), attention_code: "neutral-unrecognized-code" })));
  show();
  await screen.findByRole("heading", { name: "Detail collection" });
  const panel = screen.getByRole("region", { name: "Installation for Detail variant" });
  expect(panel).toHaveTextContent("Review workflow setup");
  expect(panel).not.toHaveTextContent("neutral-unrecognized-code");
});

it("does not attach old-revision progress to a new current revision", async () => {
  arrange({ ...progress("completed"), workflow_revision_id: "old-revision" });
  show();
  await screen.findByRole("heading", { name: "Detail collection" });
  expect(screen.queryByRole("region", { name: "Installation for Detail variant" })).not.toBeInTheDocument();
  expect(api.workflowInstallProgress).not.toHaveBeenCalled();
});

it.each([
  ["workflow-runtime-plan-changed", "The media runtime changed. Review workflow setup before continuing."],
  ["workflow-runtime-plan-unavailable", "The media runtime is unavailable. Open workflow setup to check it."],
  ["workflow-extension-review-required", "Review the extension code in Extensions. Installation continues after approval."],
] as const)("explains %s without showing its internal code", async (code, message) => {
  arrange({ ...progress("needs_attention"), attention_code: code });
  show();
  await screen.findByRole("heading", { name: "Detail collection" });
  const panel = screen.getByRole("region", { name: "Installation for Detail variant" });
  expect(panel).toHaveTextContent(message);
  expect(panel).not.toHaveTextContent(code);
  expect(api.installWorkflowOffer).not.toHaveBeenCalled();
});

it.each(["verifying", "needs_attention", "completed"] as const)(
  "keeps %s visible when a workflow needs no asset downloads", async phase => {
    arrange({ ...progress(phase), total_downloads: 0, completed_downloads: 0,
      attention_code: phase === "needs_attention" ? "workflow-runtime-plan-changed" : null });
    show();
    await screen.findByRole("heading", { name: "Detail collection" });
    const panel = screen.getByRole("region", { name: "Installation for Detail variant" });
    await waitFor(() => expect(api.workflowInstallProgress).toHaveBeenCalled());
    expect(within(panel).getByRole("status")).not.toHaveTextContent("Checking installation status");
    expect(within(panel).queryByRole("progressbar")).not.toBeInTheDocument();
    expect(panel).not.toHaveTextContent("0 of 0");
    if (phase === "needs_attention") expect(panel).toHaveTextContent("The media runtime changed.");
  },
);
