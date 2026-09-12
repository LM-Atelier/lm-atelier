import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import type { Workflow, WorkflowSummary } from "./types";

vi.mock("./api", () => ({ api: {
  workflows: vi.fn(), workflowSummaries: vi.fn(), workflow: vi.fn(), workflowFamilies: vi.fn(),
  validateWorkflow: vi.fn(), createWorkflow: vi.fn(), createWorkflowRevision: vi.fn(), updateWorkflow: vi.fn(),
} }));
vi.mock("./CustomNodesPanel", () => ({ CustomNodesPanel: () => null }));
vi.mock("./RegistryInstallsPanel", () => ({ RegistryInstallsPanel: () => null }));
vi.mock("./WorkflowRevisionReviewPanel", () => ({ WorkflowRevisionReviewPanel: () => null }));

const stamp = "2026-09-01T00:00:00Z";
const clients: QueryClient[] = [];
function detail(id: string): Workflow {
  return { id, family_id: null, name: "Workflow " + id, description: "Neutral " + id,
    operation: "text_to_image", current_revision_id: id + "-r2", revisions: [1, 2].map(version => ({
      id: id + "-r" + version, workflow_id: id, version, engine: "mock", engine_version: null,
      trusted: true, created_at: stamp, ui_graph_json: {}, input_schema_json: {},
      dependencies_json: {}, api_graph_json: { selected_graph: id + "-r" + version },
    })) };
}
function summary(id: string): WorkflowSummary {
  const value = detail(id);
  return { id: value.id, family_id: null, name: value.name, description: value.description,
    operation: value.operation, current_revision_id: value.current_revision_id,
    revision_count: 2, created_at: stamp, updated_at: stamp };
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.workflows).mockRejectedValue(new Error("Bulk graphs must not be requested"));
  vi.mocked(api.workflowSummaries).mockResolvedValue([summary("a"), summary("b")]);
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.workflow).mockImplementation(async (id) => detail(id));
});
afterEach(() => { cleanup(); clients.splice(0).forEach(client => client.clear()); });
function open() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client); render(<QueryClientProvider client={client}><WorkflowsView /></QueryClientProvider>);
  return client;
}

it("browses summaries without fetching any graph until a workflow is selected", async () => {
  open();
  expect(await screen.findByText("Workflow a")).toBeInTheDocument();
  expect(api.workflows).not.toHaveBeenCalled();
  expect(api.workflow).not.toHaveBeenCalled();
  expect(screen.getAllByText("Text to image · 2 revisions")).toHaveLength(2);
  fireEvent.click(screen.getByText("Workflow a"));
  expect(await screen.findByRole("button", { name: "New revision" })).toBeInTheDocument();
  expect(api.workflow).toHaveBeenCalledWith("a", expect.any(AbortSignal));
  expect(screen.getByText(/"selected_graph": "a-r2"/)).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Revision"), { target: { value: "a-r1" } });
  expect(screen.getByRole("button", { name: "Validate" })).toBeDisabled();
  expect(screen.getByText(/"selected_graph": "a-r1"/)).toBeInTheDocument();
});

it("does not display a late result after switching the selected workflow", async () => {
  let finish!: (value: Workflow) => void;
  vi.mocked(api.workflow).mockImplementation(async (id) => id === "a"
    ? new Promise<Workflow>(resolve => { finish = resolve; }) : detail(id));
  open(); fireEvent.click(await screen.findByText("Workflow a"));
  expect(await screen.findByText("Loading workflow details…")).toBeInTheDocument();
  await waitFor(() => expect(api.workflow).toHaveBeenCalledOnce());
  const signal = vi.mocked(api.workflow).mock.calls[0][1];
  fireEvent.click(screen.getByText("Workflow b"));
  expect(await screen.findByText(/"selected_graph": "b-r2"/)).toBeInTheDocument();
  expect(signal?.aborted).toBe(true);
  await act(async () => finish(detail("a")));
  expect(screen.queryByText(/"selected_graph": "a-r2"/)).not.toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "Workflow b" })).toBeInTheDocument();
});

it("hides stale detail after a failed refresh and retries only the selected workflow", async () => {
  const client = open(); fireEvent.click(await screen.findByText("Workflow a"));
  await screen.findByRole("button", { name: "New revision" });
  vi.mocked(api.workflow).mockRejectedValue(new Error("Selected workflow is unavailable"));
  await act(async () => { await client.invalidateQueries({ queryKey: ["workflows", "detail", "a"] }); });
  expect(await screen.findByText("Selected workflow is unavailable")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "New revision" })).not.toBeInTheDocument();
  expect(screen.queryByText(/"selected_graph": "a-r2"/)).not.toBeInTheDocument();
  expect(screen.getByText("Workflow b")).toBeInTheDocument();
  vi.mocked(api.workflow).mockResolvedValue(detail("a"));
  fireEvent.click(screen.getByRole("button", { name: "Retry workflow details" }));
  expect(await screen.findByRole("button", { name: "New revision" })).toBeInTheDocument();
  expect(api.workflows).not.toHaveBeenCalled();
});

it("preserves an open revision draft when detail refresh fails and reloads before saving", async () => {
  const client = open(); fireEvent.click(await screen.findByText("Workflow a"));
  fireEvent.click(await screen.findByRole("button", { name: "New revision" }));
  const dialog = await screen.findByRole("dialog", { name: "Create workflow revision" });
  const graph = within(dialog).getByLabelText("API-format workflow JSON");
  fireEvent.change(graph, { target: { value: '{"edited":true}' } });
  vi.mocked(api.workflow).mockRejectedValue(new Error("Selected workflow is unavailable"));
  await act(async () => { await client.invalidateQueries({ queryKey: ["workflows", "detail", "a"] }); });
  await screen.findByText("Selected workflow is unavailable");
  const save = within(dialog).getByRole("button", { name: "Create revision" });
  expect(save).toBeDisabled();
  fireEvent.click(save);
  expect(api.createWorkflow).not.toHaveBeenCalled();
  expect(api.createWorkflowRevision).not.toHaveBeenCalled();
  expect(graph).toHaveValue('{"edited":true}');
  vi.mocked(api.workflow).mockResolvedValue(detail("a"));
  fireEvent.click(within(dialog).getByRole("button", { name: "Reload workflow details" }));
  await waitFor(() => expect(save).toBeEnabled());
  expect(graph).toHaveValue('{"edited":true}');
  vi.mocked(api.createWorkflowRevision).mockResolvedValue(detail("a").revisions[1]);
  fireEvent.click(save);
  await waitFor(() => expect(api.createWorkflowRevision).toHaveBeenCalledOnce());
  expect(vi.mocked(api.createWorkflowRevision).mock.calls[0]).toEqual([
    "a", expect.objectContaining({ api_graph: { edited: true } }),
  ]);
  expect(api.createWorkflow).not.toHaveBeenCalled();
});
