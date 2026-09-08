import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { WorkflowFamilyArchive } from "./WorkflowFamilyArchive";
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
function impact(overrides: Partial<WorkflowFamilyRemovalImpact> = {}): WorkflowFamilyRemovalImpact {
  return { family_id: "family-b", removal_strategy: "archive", archive_blocked: false,
    revision_count: 2, current_revision_count: 1, chat_selection_count: 0, project_selection_count: 0,
    project_revision_pin_count: 0, active_run_count: 0, queued_step_count: 0, historical_run_count: 0,
    active_activation_count: 0, default_for: [], dependencies: [], ...overrides };
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
  vi.mocked(api.workflowFamilyRemovalImpact).mockResolvedValue(impact());
  vi.mocked(api.updateWorkflowFamily).mockResolvedValue({ ...family("b"), archived: true, enabled: false });
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });

describe("archiving workflow families from the page", () => {
  it("opens the selected family, confirms its impact, and refreshes the consuming surfaces", async () => {
    vi.mocked(api.workflowFamilyRemovalImpact).mockResolvedValue(impact({ queued_step_count: 1, active_run_count: 1 }));
    const client = wrap(<WorkflowsView />);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    fireEvent.click(await screen.findByText("Workflow b"));
    fireEvent.click(screen.getByRole("button", { name: "Archive family" }));
    const dialog = await screen.findByRole("dialog", { name: "Archive Family b?" });
    expect(await within(dialog).findByText("1 queued step still runs")).toBeInTheDocument();
    expect(within(dialog).getByText("1 run in progress finishes")).toBeInTheDocument();
    expect(within(dialog).queryByText("Cannot be undone")).not.toBeInTheDocument();
    expect(api.workflowFamilyRemovalImpact).toHaveBeenCalledWith("family-b");
    expect(api.updateWorkflowFamily).not.toHaveBeenCalled();
    fireEvent.click(within(dialog).getByRole("button", { name: "Archive it" }));
    await waitFor(() => expect(api.updateWorkflowFamily).toHaveBeenCalledWith("family-b", { archived: true }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    for (const key of ["workflows", "workflow-families", "studio-capabilities"]) {
      expect(invalidate).toHaveBeenCalledWith({ queryKey: [key] });
    }
  });

  it("keeps confirmation disabled until the impact arrives and cancels without a write", async () => {
    let resolve: (value: WorkflowFamilyRemovalImpact) => void = () => { throw new Error("uninitialized"); };
    vi.mocked(api.workflowFamilyRemovalImpact).mockReturnValue(new Promise((done) => { resolve = done; }));
    const close = vi.fn();
    wrap(<WorkflowFamilyArchive family={family("b")} onClose={close} />);
    expect(screen.getByRole("button", { name: "Archive" })).toBeDisabled();
    expect(close).not.toHaveBeenCalled();
    await act(async () => resolve(impact()));
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(close).toHaveBeenCalledOnce();
    expect(api.updateWorkflowFamily).not.toHaveBeenCalled();
  });

  it("explains the selected/default guard instead of telling an idle user to drain a queue", async () => {
    vi.mocked(api.workflowFamilyRemovalImpact).mockResolvedValue(impact({ archive_blocked: true, default_for: ["image"] }));
    wrap(<WorkflowFamilyArchive family={family("b")} onClose={vi.fn()} />);
    await screen.findByRole("dialog", { name: "Family b is still in use" });
    expect(screen.getByText(/selected by a chat or project, or is set as a default/i)).toBeInTheDocument();
    expect(screen.queryByText(/let the queue drain/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Archive it" })).not.toBeInTheDocument();
    expect(api.updateWorkflowFamily).not.toHaveBeenCalled();
  });

  it("retains the confirmation and shows a refused write, then allows a retry", async () => {
    vi.mocked(api.updateWorkflowFamily).mockRejectedValueOnce(new Error("The family became selected"));
    wrap(<WorkflowsView />);
    fireEvent.click(await screen.findByText("Workflow b"));
    fireEvent.click(screen.getByRole("button", { name: "Archive family" }));
    fireEvent.click(await screen.findByRole("button", { name: "Archive it" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The family became selected");
    expect(screen.getByRole("dialog", { name: "Archive Family b?" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Archive it" }));
    await waitFor(() => expect(api.updateWorkflowFamily).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("does not offer family archival for an ungrouped workflow", async () => {
    vi.mocked(api.workflowFamilies).mockResolvedValue([]);
    wrap(<WorkflowsView />);
    fireEvent.click(await screen.findByText("Workflow a"));
    expect(screen.getByRole("button", { name: "New revision" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Archive family" })).not.toBeInTheDocument();
  });
});
