import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import type { Workflow, WorkflowFamily, WorkflowFamilyRemovalImpact } from "./types";

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

describe("workflow family usage", () => {
  it("loads the existing usage report only when requested and shows exact counts", async () => {
    vi.mocked(api.workflowFamilyRemovalImpact).mockResolvedValue(impact("a"));
    wrap(<WorkflowsView />);
    fireEvent.click(await screen.findByText("Workflow a"));
    const toggle = screen.getByRole("button", { name: "Show usage" });
    expect(api.workflowFamilyRemovalImpact).not.toHaveBeenCalled();
    fireEvent.click(toggle);
    const region = screen.getByRole("region", { name: "Family usage" });
    await within(region).findByText("Chat selections");
    for (const [label, count] of [["Chat selections", 2], ["Project selections", 1], ["Pinned project revisions", 4], ["Active runs", 1], ["Queued steps", 3], ["Historical runs", 12]] as const) {
      expect(within(region).getByText(label).nextElementSibling).toHaveTextContent(String(count));
    }
    expect(api.workflowFamilyRemovalImpact).toHaveBeenCalledWith("family-a");
    expect(screen.getByRole("button", { name: "New revision" })).toBeInTheDocument();
  });

  it("refreshes cached usage before enabling archive confirmation", async () => {
    vi.mocked(api.workflowFamilyRemovalImpact).mockResolvedValueOnce(impact("a"))
      .mockReturnValueOnce(new Promise(() => {}));
    wrap(<WorkflowsView />);
    fireEvent.click(await screen.findByText("Workflow a"));
    fireEvent.click(screen.getByRole("button", { name: "Show usage" }));
    await screen.findByText("Historical runs");
    fireEvent.click(screen.getByRole("button", { name: "Archive family" }));
    await waitFor(() => expect(api.workflowFamilyRemovalImpact).toHaveBeenCalledTimes(2));
    const dialog = screen.getByRole("dialog", { name: "Archive Family a?" });
    expect(within(dialog).getByRole("button", { name: /^Archive(?: it)?$/ })).toBeDisabled();
    expect(api.updateWorkflowFamily).not.toHaveBeenCalled();
  });

  it("shows loading rather than a false empty report", async () => {
    vi.mocked(api.workflowFamilyRemovalImpact).mockReturnValue(new Promise(() => {}));
    wrap(<WorkflowsView />);
    fireEvent.click(await screen.findByText("Workflow a"));
    fireEvent.click(screen.getByRole("button", { name: "Show usage" }));
    expect(await screen.findByText("Loading usage…")).toBeInTheDocument();
    expect(screen.queryByText("Chat selections")).not.toBeInTheDocument();
  });

  it("preserves a failed report as an error and retries the same family", async () => {
    vi.mocked(api.workflowFamilyRemovalImpact).mockRejectedValueOnce(new Error("Usage could not be read"))
      .mockResolvedValueOnce(impact("a"));
    wrap(<WorkflowsView />);
    fireEvent.click(await screen.findByText("Workflow a"));
    fireEvent.click(screen.getByRole("button", { name: "Show usage" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Usage could not be read");
    fireEvent.click(screen.getByRole("button", { name: "Retry usage" }));
    await screen.findByText("Historical runs");
    expect(api.workflowFamilyRemovalImpact).toHaveBeenNthCalledWith(2, "family-a");
  });

  it("does not show a prior family's delayed report after selection changes", async () => {
    let finish: ((value: WorkflowFamilyRemovalImpact) => void) | undefined;
    vi.mocked(api.workflowFamilyRemovalImpact).mockImplementation((id) => id === "family-a"
      ? new Promise((resolve) => { finish = resolve; }) : Promise.resolve({ ...impact("b"), historical_run_count: 27 }));
    wrap(<WorkflowsView />);
    fireEvent.click(await screen.findByText("Workflow a"));
    fireEvent.click(screen.getByRole("button", { name: "Show usage" }));
    await waitFor(() => expect(api.workflowFamilyRemovalImpact).toHaveBeenCalledWith("family-a"));
    fireEvent.click(screen.getByText("Workflow b"));
    fireEvent.click(screen.getByRole("button", { name: "Show usage" }));
    await screen.findByText("27");
    await act(async () => finish?.({ ...impact("a"), historical_run_count: 99 }));
    expect(screen.queryByText("99")).not.toBeInTheDocument();
    expect(screen.getByText("Historical runs").nextElementSibling).toHaveTextContent("27");
  });
});
