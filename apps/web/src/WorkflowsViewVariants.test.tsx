import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import type { Workflow, WorkflowFamily } from "./types";

vi.mock("./api", () => ({ api: {
  workflows: vi.fn(), workflowFamilies: vi.fn(), workflowFamilyRemovalImpact: vi.fn(),
  updateWorkflowFamily: vi.fn(), setWorkflowFamilyPreference: vi.fn(),
} }));
vi.mock("./CustomNodesPanel", () => ({ CustomNodesPanel: () => null }));
vi.mock("./RegistryInstallsPanel", () => ({ RegistryInstallsPanel: () => null }));
vi.mock("./WorkflowRevisionReviewPanel", () => ({ WorkflowRevisionReviewPanel: () => null }));

function workflow(id: string): Workflow {
  return { id, name: `Workflow ${id}`, description: "Neutral fixture", operation: "text_to_image",
    current_revision_id: `revision-${id}`, revisions: [{ id: `revision-${id}`, workflow_id: id,
      version: 1, engine: "comfyui", engine_version: null, ui_graph_json: {}, api_graph_json: {},
      input_schema_json: {}, dependencies_json: {}, trusted: false, created_at: "2026-09-08T00:00:00Z" }] };
}
function family(id: string): WorkflowFamily {
  return { id: `family-${id}`, name: `Family ${id}`, description: "", use_case: "", tags: [],
    enabled: true, archived: false, compatibility: false, created_at: "2026-09-08T00:00:00Z",
    updated_at: "2026-09-08T00:00:00Z", preferences: [], variants: [{ id, variant_key: "image",
      name: `Workflow ${id}`, operation: "text_to_image", current_revision_id: `revision-${id}`,
      current_revision_version: 1, engine: "comfyui", capabilities: ["image"], trusted: false,
      readiness: "review_required", readiness_reason: "revision_untrusted" }] };
}
const clients: QueryClient[] = [];
function wrap(element: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}>{element}</QueryClientProvider>);
  return client;
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.workflows).mockResolvedValue([workflow("a"), workflow("b")]);
  vi.mocked(api.workflowFamilies).mockResolvedValue([family("a"), family("b")]);

  vi.mocked(api.updateWorkflowFamily).mockResolvedValue({ ...family("b"), archived: true, enabled: false });
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });

describe("workflow family variants", () => {
  it("shows each current variant and explains its server readiness", async () => {
    const mixed = family("a");
    mixed.variants.push(
      { ...mixed.variants[0], id: "video", name: "Motion", operation: "text_to_video",
        current_revision_id: "motion-v3", current_revision_version: 3,
        readiness: "setup_required", readiness_reason: "activation_not_ready" },
      { ...mixed.variants[0], id: "ready", name: "Ready image", readiness: "ready", readiness_reason: null },
      { ...mixed.variants[0], id: "missing", name: "Missing revision", current_revision_id: null,
        current_revision_version: null, readiness: "setup_required", readiness_reason: "current_revision_missing" },
    );
    vi.mocked(api.workflowFamilies).mockResolvedValue([mixed]);
    wrap(<WorkflowsView />);
    fireEvent.click(await screen.findByText("Workflow a"));
    fireEvent.click(screen.getByRole("button", { name: "Show operation variants" }));
    const region = within(screen.getByRole("region", { name: "Operation variants" }));
    expect(region.getAllByRole("listitem")).toHaveLength(4);
    expect(region.getByText("Text to video")).toBeInTheDocument();
    expect(region.getByText("Current revision: v3")).toBeInTheDocument();
    expect(region.getByText("This revision needs review before it can run.")).toBeInTheDocument();
    expect(region.getByText("Its required dependencies are not activated for this revision.")).toBeInTheDocument();
    expect(region.getByText("Ready to run.")).toBeInTheDocument();
    expect(region.getByText("No current revision")).toBeInTheDocument();
    expect(region.queryByRole("button", { name: /install/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "New revision" })).toBeInTheDocument();
  });

  it("resets the disclosure when selecting another family", async () => {
    wrap(<WorkflowsView />);
    fireEvent.click(await screen.findByText("Workflow a"));
    fireEvent.click(screen.getByRole("button", { name: "Show operation variants" }));
    fireEvent.click(screen.getByText("Workflow b"));
    expect(screen.getByRole("button", { name: "Show operation variants" })).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(screen.getByRole("button", { name: "Show operation variants" }));
    const region = within(screen.getByRole("region", { name: "Operation variants" }));
    expect(region.getByText("Workflow b")).toBeInTheDocument();
    expect(region.queryByText("Workflow a")).not.toBeInTheDocument();
    expect(screen.getAllByRole("region", { name: "Family usage" })).toHaveLength(1);
    expect(screen.getAllByRole("region", { name: "Family dependencies" })).toHaveLength(1);
  });

  it("keeps an unknown readiness reason neutral", async () => {
    const blocked = family("a");
    blocked.variants[0] = { ...blocked.variants[0], readiness: "unavailable", readiness_reason: "future_server_reason" };
    vi.mocked(api.workflowFamilies).mockResolvedValue([blocked]);
    wrap(<WorkflowsView />);
    fireEvent.click(await screen.findByText("Workflow a"));
    fireEvent.click(screen.getByRole("button", { name: "Show operation variants" }));
    const region = within(screen.getByRole("region", { name: "Operation variants" }));
    expect(region.getByText("Unavailable")).toBeInTheDocument();
    expect(region.getByText("No further readiness details are available.")).toBeInTheDocument();
    expect(region.queryByText("future_server_reason")).not.toBeInTheDocument();
  });
});
