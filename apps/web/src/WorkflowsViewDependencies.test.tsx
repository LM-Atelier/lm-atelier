import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import type { Workflow, WorkflowFamily, WorkflowFamilyRemovalImpact, WorkflowDependencyImpact } from "./types";

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

function impact(id: string): WorkflowFamilyRemovalImpact {
  return { family_id: `family-${id}`, removal_strategy: "archive", archive_blocked: false,
    revision_count: 3, current_revision_count: 1, chat_selection_count: 2, project_selection_count: 1,
    project_revision_pin_count: 4, active_run_count: 1, queued_step_count: 3, historical_run_count: 12,
    active_activation_count: 1, default_for: [], dependencies: [] };
}

function dependency(overrides: Partial<WorkflowDependencyImpact> = {}): WorkflowDependencyImpact {
  return { resource_kind: "model_install", resource_id: "model-a", resource_name: "Neutral image model",
    binding_count: 3, revision_count: 2, current_revision: true, shared: true,
    other_workflow_count: 2, other_family_ids: ["family-c"], ...overrides };
}
async function openDependencies() {
  fireEvent.click(await screen.findByText("Workflow a"));
  fireEvent.click(screen.getByRole("button", { name: "Show dependencies" }));
  return screen.getByRole("region", { name: "Family dependencies" });
}

describe("workflow family dependencies", () => {
  it("loads on demand and groups recorded resources by their declared kinds", async () => {
    const dependencies = [dependency(), ...([
      ["model_profile", "Neutral profile"], ["model_asset", "Neutral adapter"],
      ["custom_node", "Neutral node"], ["registry_package", "Neutral package"], ["runtime", "Neutral runtime"],
    ] as const).map(([resource_kind, resource_name]) => dependency({ resource_kind, resource_name,
      resource_id: resource_kind, binding_count: 1, revision_count: 1, shared: false, other_workflow_count: 0 }))];
    vi.mocked(api.workflowFamilyRemovalImpact).mockResolvedValue({ ...impact("a"), dependencies });
    wrap(<WorkflowsView />);
    fireEvent.click(await screen.findByText("Workflow a"));
    expect(api.workflowFamilyRemovalImpact).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Show dependencies" }));
    const region = screen.getByRole("region", { name: "Family dependencies" });
    await within(region).findByText("Neutral image model");
    for (const name of ["Model profiles", "Model files", "Model assets", "Custom nodes", "Registry packages", "Runtimes"]) {
      expect(within(region).getByRole("heading", { name })).toBeInTheDocument();
    }
    expect(within(region).getByText("3 bindings across 2 revisions")).toBeInTheDocument();
    expect(within(region).getByText("Also used by 2 other workflows")).toBeInTheDocument();
    expect(api.workflowFamilyRemovalImpact).toHaveBeenCalledWith("family-a");
    expect(screen.getByRole("button", { name: "New revision" })).toBeInTheDocument();
  });

  it("distinguishes earlier-revision bindings and filters current revisions without refetching", async () => {
    vi.mocked(api.workflowFamilyRemovalImpact).mockResolvedValue({ ...impact("a"), dependencies: [
      dependency(), dependency({ resource_id: "older-model", resource_name: "Earlier model", current_revision: false }),
    ] });
    wrap(<WorkflowsView />);
    const region = await openDependencies();
    await within(region).findByText("Earlier model");
    expect(within(region).getByText("Earlier revisions only")).toBeInTheDocument();
    fireEvent.click(within(region).getByRole("checkbox", { name: "Current revisions only" }));
    expect(within(region).queryByText("Earlier model")).not.toBeInTheDocument();
    expect(within(region).getByText("Neutral image model")).toBeInTheDocument();
    expect(api.workflowFamilyRemovalImpact).toHaveBeenCalledTimes(1);
    fireEvent.click(within(region).getByRole("checkbox", { name: "Current revisions only" }));
    expect(within(region).getByText("Earlier model")).toBeInTheDocument();
  });

  it("does not describe an unresolved request as an empty dependency inventory", async () => {
    vi.mocked(api.workflowFamilyRemovalImpact).mockReturnValue(new Promise(() => {}));
    wrap(<WorkflowsView />);
    const region = await openDependencies();
    expect(await within(region).findByRole("status")).toHaveTextContent("Loading dependencies");
    expect(within(region).queryByText("No recorded dependency bindings.")).not.toBeInTheDocument();
  });

  it("keeps read failures visible and retries before reporting a genuinely empty inventory", async () => {
    vi.mocked(api.workflowFamilyRemovalImpact).mockRejectedValueOnce(new Error("Dependency report unavailable"))
      .mockResolvedValueOnce(impact("a"));
    wrap(<WorkflowsView />);
    const region = await openDependencies();
    expect(await within(region).findByRole("alert")).toHaveTextContent("Dependency report unavailable");
    expect(within(region).queryByText("No recorded dependency bindings.")).not.toBeInTheDocument();
    fireEvent.click(within(region).getByRole("button", { name: "Retry dependencies" }));
    await within(region).findByText("No recorded dependency bindings.");
    expect(api.workflowFamilyRemovalImpact).toHaveBeenNthCalledWith(2, "family-a");
  });

  it("does not carry a previous family's expansion or delayed report into the next family", async () => {
    let finish: ((value: WorkflowFamilyRemovalImpact) => void) | undefined;
    vi.mocked(api.workflowFamilyRemovalImpact).mockImplementation((id) => id === "family-a"
      ? new Promise((resolve) => { finish = resolve; })
      : Promise.resolve({ ...impact("b"), dependencies: [dependency({ resource_name: "Second family model" })] }));
    wrap(<WorkflowsView />);
    await openDependencies();
    await waitFor(() => expect(api.workflowFamilyRemovalImpact).toHaveBeenCalledWith("family-a"));
    fireEvent.click(screen.getByText("Workflow b"));
    expect(screen.getByRole("button", { name: "Show dependencies" })).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(screen.getByRole("button", { name: "Show dependencies" }));
    await screen.findByText("Second family model");
    await act(async () => { finish?.({ ...impact("a"), dependencies: [dependency()] }); });
    expect(screen.queryByText("Neutral image model")).not.toBeInTheDocument();
    expect(screen.getByText("Second family model")).toBeInTheDocument();
  });
});
