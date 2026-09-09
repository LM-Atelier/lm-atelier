import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
  return { id, family_id: `family-${id}`, name: `Workflow ${id}`, description: "Neutral fixture", operation: "text_to_image",
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

describe("browsing workflow families", () => {
  it("filters the family source without guessing the origin of ungrouped definitions", async () => {
    vi.mocked(api.workflowFamilies).mockResolvedValue([
      { ...family("a"), compatibility: true }, family("b"),
    ]);
    vi.mocked(api.workflows).mockResolvedValue([
      workflow("a"), workflow("b"), { ...workflow("c"), family_id: null },
    ]);
    wrap(<WorkflowsView />);
    await screen.findByRole("heading", { name: "Family a" });
    expect(screen.getByRole("heading", { name: "Ungrouped workflows" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Family source" }), { target: { value: "profile" } });
    expect(screen.getByRole("heading", { name: "Family a" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Family b" })).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Ungrouped workflows" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Family source" }), { target: { value: "workflow" } });
    expect(screen.queryByRole("heading", { name: "Family a" })).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Family b" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Ungrouped workflows" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Family source" }), { target: { value: "" } });
    expect(screen.getByRole("heading", { name: "Ungrouped workflows" })).toBeInTheDocument();
    fireEvent.click(screen.getByText("Workflow c"));
    expect(screen.getByRole("button", { name: "New revision" })).toBeInTheDocument();
  });

  it("sorts readiness using only the variants matching the operation filter", async () => {
    const alpha = { ...family("a"), name: "Alpha family" };
    alpha.variants[0] = { ...alpha.variants[0], readiness: "unavailable" };
    alpha.variants.push({ ...family("c").variants[0], operation: "text_to_video", readiness: "ready" });
    const beta = { ...family("b"), name: "Beta family" };
    beta.variants[0] = { ...beta.variants[0], readiness: "setup_required" };
    vi.mocked(api.workflowFamilies).mockResolvedValue([beta, alpha]);
    wrap(<WorkflowsView />);
    await screen.findByRole("heading", { name: "Alpha family" });
    const names = () => screen.getAllByRole("heading", { level: 3 })
      .filter((heading) => heading.id.startsWith("workflow-family-"))
      .map((heading) => heading.textContent);
    expect(names()).toEqual(["Alpha family", "Beta family"]);
    fireEvent.change(screen.getByRole("combobox", { name: "Sort workflow families" }), { target: { value: "readiness" } });
    expect(names()).toEqual(["Alpha family", "Beta family"]);
    fireEvent.change(screen.getByRole("combobox", { name: "Filter by operation" }), { target: { value: "text_to_image" } });
    expect(names()).toEqual(["Beta family", "Alpha family"]);
    fireEvent.change(screen.getByRole("combobox", { name: "Sort workflow families" }), { target: { value: "name" } });
    expect(names()).toEqual(["Alpha family", "Beta family"]);
  });

  it("groups variants by family and searches family metadata without losing the revision editor", async () => {
    vi.mocked(api.workflowFamilies).mockResolvedValue([
      { ...family("a"), name: "Landscape family", use_case: "Outdoor scenes" },
      { ...family("b"), name: "Portrait family", use_case: "Headshots" },
    ]);
    wrap(<WorkflowsView />);
    expect(await screen.findByRole("heading", { name: "Landscape family" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Portrait family" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("searchbox", { name: "Search workflow families" }), { target: { value: "headshots" } });
    expect(screen.queryByRole("heading", { name: "Landscape family" })).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Portrait family" })).toBeInTheDocument();
    fireEvent.click(screen.getByText("Workflow b"));
    expect(screen.getByRole("button", { name: "New revision" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export" })).toBeInTheDocument();
  });

  it("filters by operation, readiness and defaults using the actual variant facts", async () => {
    const image = family("a");
    const video = family("b");
    image.variants[0] = { ...image.variants[0], readiness: "ready", readiness_reason: null };
    image.preferences = [{ selector_capability: "image", enabled: true, is_default: true, sort_order: 0 }];
    video.variants[0] = { ...video.variants[0], operation: "text_to_video", readiness: "setup_required", readiness_reason: "activation_not_ready" };
    vi.mocked(api.workflowFamilies).mockResolvedValue([video, image]);
    wrap(<WorkflowsView />);
    await screen.findByRole("heading", { name: "Family a" });
    expect(screen.getByText("Ready", { selector: ".badge" })).toBeInTheDocument();
    expect(screen.getByText("Needs setup", { selector: ".badge" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Filter by operation" }), { target: { value: "text_to_video" } });
    expect(screen.queryByRole("heading", { name: "Family a" })).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Family b" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Filter by operation" }), { target: { value: "" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Filter by readiness" }), { target: { value: "ready" } });
    expect(screen.queryByRole("heading", { name: "Family b" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Filter by readiness" }), { target: { value: "" } });
    fireEvent.click(screen.getByRole("checkbox", { name: "Defaults only" }));
    expect(screen.queryByRole("heading", { name: "Family b" })).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Family a" })).toBeInTheDocument();
  });

  it("requires combined filters to match the same variant", async () => {
    const mixed = family("a");
    mixed.variants[0] = { ...mixed.variants[0], readiness: "ready", readiness_reason: null };
    mixed.variants.push({ ...family("b").variants[0], operation: "text_to_video", readiness: "setup_required" });
    vi.mocked(api.workflowFamilies).mockResolvedValue([mixed]);
    wrap(<WorkflowsView />);
    await screen.findByRole("heading", { name: "Family a" });
    fireEvent.change(screen.getByRole("combobox", { name: "Filter by operation" }), { target: { value: "text_to_video" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Filter by readiness" }), { target: { value: "ready" } });
    expect(screen.queryByRole("heading", { name: "Family a" })).not.toBeInTheDocument();
    expect(screen.getByText("No workflows match these filters.")).toBeInTheDocument();
  });

  it("re-queries archived families on the server and restores the ordinary list when unchecked", async () => {
    vi.mocked(api.workflowFamilies).mockImplementation(async (_capability, includeArchived) => includeArchived
      ? [family("a"), { ...family("b"), name: "Archived family", archived: true, enabled: false }]
      : [family("a")]);
    wrap(<WorkflowsView />);
    await screen.findByRole("heading", { name: "Family a" });
    expect(screen.queryByRole("heading", { name: "Archived family" })).not.toBeInTheDocument();
    expect(screen.queryByText("Workflow b")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "Show archived families" }));
    expect(await screen.findByRole("heading", { name: "Archived family" })).toBeInTheDocument();
    expect(api.workflowFamilies).toHaveBeenCalledWith(undefined, true);
    fireEvent.click(screen.getByRole("checkbox", { name: "Show archived families" }));
    await waitFor(() => expect(screen.queryByRole("heading", { name: "Archived family" })).not.toBeInTheDocument());
  });

  it("keeps an ungrouped legacy workflow selectable and gives an explicit empty search result", async () => {
    vi.mocked(api.workflows).mockResolvedValue([workflow("a"), { ...workflow("b"), family_id: null }]);
    vi.mocked(api.workflowFamilies).mockResolvedValue([family("a")]);
    wrap(<WorkflowsView />);
    await screen.findByRole("heading", { name: "Ungrouped workflows" });
    fireEvent.click(screen.getByText("Workflow b"));
    expect(screen.getByRole("button", { name: "New revision" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("searchbox", { name: "Search workflow families" }), { target: { value: "no matching family" } });
    expect(screen.getByText("No workflows match these filters.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "New revision" })).toBeInTheDocument();
  });
});
