import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import type { Job, Workflow, WorkflowFamily, WorkflowInstallOffer } from "./types";

vi.mock("./api", () => ({ api: {
  workflows: vi.fn(), workflowFamilies: vi.fn(), workflowFamilyRemovalImpact: vi.fn(),
  updateWorkflowFamily: vi.fn(), setWorkflowFamilyPreference: vi.fn(), installWorkflowOffer: vi.fn(),
} }));
vi.mock("./CustomNodesPanel", () => ({ CustomNodesPanel: () => null }));
vi.mock("./RegistryInstallsPanel", () => ({ RegistryInstallsPanel: () => null }));
vi.mock("./WorkflowRevisionReviewPanel", () => ({ WorkflowRevisionReviewPanel: () => null }));

function offer(): WorkflowInstallOffer {
  return { id: "offer-review-a", workflow_revision_id: "revision-a",
    workflow_artifact_sha256: "a".repeat(64), dependency_contract_sha256: "b".repeat(64),
    binding_plan_sha256: "c".repeat(64), offer_sha256: "d".repeat(64), plan_count: 1,
    total_bytes: 17, status: "ready", queued_at: null, completed_at: null,
    invalidated_at: null, invalidation_code: null, invalidation_reason: null,
    assets: [{ reference_filename: "styles/detail.safetensors", kind: "lora",
      install_plan_id: "plan-a", install_plan_hash: "e".repeat(64), provider: "civitai",
      remote_id: "101", revision: "202", artifact_path: "detail.safetensors",
      artifact_kind: "lora", target_folder: "loras", size_bytes: 17, sha256: "f".repeat(64) }] };
}
function workflow(): Workflow {
  return { id: "a", family_id: "family-a", name: "Detail variant", description: "Neutral fixture",
    operation: "text_to_image", current_revision_id: "revision-a",
    revisions: [{ id: "revision-a", workflow_id: "a", version: 1, engine: "comfyui",
      engine_version: null, ui_graph_json: {}, api_graph_json: {}, input_schema_json: {},
      dependencies_json: {}, trusted: true, created_at: "2026-09-10T00:00:00Z" }] };
}
function family(): WorkflowFamily {
  return { id: "family-a", name: "Detail collection", description: "", use_case: "", tags: [],
    enabled: true, archived: false, compatibility: false, created_at: "2026-09-10T00:00:00Z",
    updated_at: "2026-09-10T00:00:00Z", preferences: [], variants: [{ id: "a", variant_key: "image",
      name: "Detail variant", operation: "text_to_image", current_revision_id: "revision-a",
      current_revision_version: 1, engine: "comfyui", capabilities: ["image"], trusted: true,
      readiness: "setup_required", readiness_reason: "activation_not_ready",
      setup_resolution: "reviewed_download_available", install_offer: offer() }] };
}
const queued: Job = { id: "download-a", kind: "download", status: "queued", run_id: null,
  progress: 0, phase: "queued", payload_json: {}, result_json: {}, error: null, attempt: 0,
  cancellable: true, created_at: "2026-09-10T00:00:00Z", updated_at: "2026-09-10T00:00:00Z",
  started_at: null, completed_at: null };
const clients: QueryClient[] = [];
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><WorkflowsView /></QueryClientProvider>);
  return client;
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.workflows).mockResolvedValue([workflow()]);
  vi.mocked(api.workflowFamilies).mockResolvedValue([family()]);
  vi.mocked(api.installWorkflowOffer).mockResolvedValue([queued]);
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });

it.each(["card", "details"])("reviews an exact download from the %s before sending its opaque ID", async (location) => {
  const client = show();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  await screen.findByRole("heading", { name: "Detail collection" });
  let area: HTMLElement = document.body;
  if (location === "details") {
    fireEvent.click(screen.getByRole("button", { name: /^Detail variant Text to image/ }));
    fireEvent.click(screen.getByRole("button", { name: "Show operation variants" }));
    area = screen.getByRole("region", { name: "Operation variants" });
  }
  fireEvent.click(within(area).getByRole("button", { name: "Review downloads for Detail variant" }));
  const dialog = screen.getByRole("dialog", { name: "Review workflow downloads" });
  expect(within(dialog).getByText("styles/detail.safetensors")).toBeInTheDocument();
  expect(within(dialog).getAllByText(/17 B/).length).toBeGreaterThan(0);
  expect(within(dialog).queryByText("plan-a")).not.toBeInTheDocument();
  expect(api.installWorkflowOffer).not.toHaveBeenCalled();
  fireEvent.click(within(dialog).getByRole("button", { name: "Download reviewed files" }));
  await waitFor(() => expect(api.installWorkflowOffer).toHaveBeenCalledExactlyOnceWith("offer-review-a"));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(screen.getByRole("status")).toHaveTextContent("Downloads queued");
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["workflow-families"] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["jobs"] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["workflows"] });
});

it("retains the reviewed files and refusal when the server rejects a changed offer", async () => {
  vi.mocked(api.installWorkflowOffer).mockRejectedValue(new Error("The reviewed download changed."));
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Review downloads for Detail variant" }));
  fireEvent.click(screen.getByRole("button", { name: "Download reviewed files" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("The reviewed download changed.");
  expect(screen.getByRole("dialog")).toHaveTextContent("styles/detail.safetensors");
  expect(screen.queryByText(/Downloads queued/)).not.toBeInTheDocument();
});

it("keeps a pending request bound to one review and prevents duplicate submissions", async () => {
  let finish: ((jobs: Job[]) => void) | undefined;
  vi.mocked(api.installWorkflowOffer).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Review downloads for Detail variant" }));
  const download = screen.getByRole("button", { name: "Download reviewed files" });
  fireEvent.click(download);
  await waitFor(() => expect(download).toBeDisabled());
  fireEvent.click(download);
  fireEvent.click(screen.getByRole("button", { name: "Close download review" }));
  expect(screen.getByRole("dialog")).toBeInTheDocument();
  expect(api.installWorkflowOffer).toHaveBeenCalledTimes(1);
  await act(async () => { finish?.([queued]); });
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
});

it.each(["no-offer", "attention", "ready", "queued", "wrong-revision", "disabled"])(
  "offers no download action for %s state", async (scenario) => {
    const value = family();
    const variant = value.variants[0];
    if (scenario === "no-offer") variant.install_offer = null;
    else if (scenario === "attention") variant.setup_resolution = "attention_required";
    else if (scenario === "ready") variant.readiness = "ready";
    else if (scenario === "disabled") value.enabled = false;
    else if (variant.install_offer) {
      if (scenario === "queued") variant.install_offer.status = "queued";
      else variant.install_offer.workflow_revision_id = "other-revision";
    }
    vi.mocked(api.workflowFamilies).mockResolvedValue([value]);
    show();
    await screen.findByRole("heading", { name: "Detail collection" });
    expect(screen.queryByRole("button", { name: "Review downloads for Detail variant" })).not.toBeInTheDocument();
    expect(api.installWorkflowOffer).not.toHaveBeenCalled();
  },
);
