import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import type { Workflow, WorkflowFamily } from "./types";

vi.mock("./api", () => ({ api: { workflows: vi.fn(), workflowFamilies: vi.fn() } }));
vi.mock("./CustomNodesPanel", () => ({ CustomNodesPanel: () => null }));
vi.mock("./RegistryInstallsPanel", () => ({ RegistryInstallsPanel: () => null }));
vi.mock("./WorkflowRevisionReviewPanel", () => ({ WorkflowRevisionReviewPanel: () => null }));
function workflow(id: string): Workflow {
  return { id, family_id: `family-${id}`, name: `Workflow ${id}`, description: "",
    operation: "text_to_image", current_revision_id: `revision-${id}`,
    revisions: [{ id: `revision-${id}`, workflow_id: id, version: 1, engine: "mock",
      engine_version: null, ui_graph_json: {}, api_graph_json: {}, input_schema_json: {},
      dependencies_json: {}, trusted: true, created_at: "2026-09-10T00:00:00Z" }] };
}
function family(id: string, summary?: { dependency_count: number; names: string[] } | null): WorkflowFamily {
  return { id: `family-${id}`, name: `Family ${id}`, description: "", use_case: "",
    tags: [], enabled: true, archived: false, compatibility: false,
    created_at: "2026-09-10T00:00:00Z", updated_at: "2026-09-10T00:00:00Z",
    ...(summary !== undefined ? { dependency_summary: summary } : {}), preferences: [],
    variants: [{ id, variant_key: "image", name: `Workflow ${id}`, operation: "text_to_image",
      current_revision_id: `revision-${id}`, current_revision_version: 1, engine: "mock",
      capabilities: [], trusted: true, readiness: "ready", readiness_reason: null }] };
}
const clients: QueryClient[] = [];
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><WorkflowsView /></QueryClientProvider>);
  return client;
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.workflows).mockResolvedValue([workflow("a"), workflow("b"), workflow("c")]);
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });

it("requests dependency summaries and finds a family by a current dependency name", async () => {
  vi.mocked(api.workflowFamilies).mockResolvedValue([
    family("a", { dependency_count: 2, names: ["Landscape checkpoint", "Encoder"] }),
    family("b", { dependency_count: 1, names: ["Other checkpoint"] }),
  ]);
  show();
  await screen.findByRole("heading", { name: "Family a" });
  expect(api.workflowFamilies).toHaveBeenCalledWith(undefined, false, true);
  expect(within(screen.getByRole("region", { name: "Family a" })).getByText("2 recorded dependencies")).toBeInTheDocument();
  fireEvent.change(screen.getByRole("searchbox", { name: "Search workflow families" }),
    { target: { value: " LANDSCAPE CHECKPOINT " } });
  expect(screen.getByRole("heading", { name: "Family a" })).toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "Family b" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByText("Workflow a"));
  expect(screen.getByRole("button", { name: "New revision" })).toBeInTheDocument();
});

it("shows exact empty and single-dependency counts without treating an unknown summary as empty", async () => {
  vi.mocked(api.workflowFamilies).mockResolvedValue([
    family("a", { dependency_count: 0, names: [] }), family("b", { dependency_count: 1, names: ["Encoder"] }),
    family("c"),
  ]);
  show();
  await screen.findByRole("heading", { name: "Family a" });
  expect(within(screen.getByRole("region", { name: "Family a" })).getByText("0 recorded dependencies")).toBeInTheDocument();
  expect(within(screen.getByRole("region", { name: "Family b" })).getByText("1 recorded dependency")).toBeInTheDocument();
  expect(within(screen.getByRole("region", { name: "Family c" })).queryByText(/recorded dependenc/)).not.toBeInTheDocument();
});

it("drops a dependency-name match after the current family summary changes", async () => {
  let current = family("a", { dependency_count: 1, names: ["Previous checkpoint"] });
  vi.mocked(api.workflowFamilies).mockImplementation(async () => [current]);
  const client = show();
  await screen.findByRole("heading", { name: "Family a" });
  fireEvent.change(screen.getByRole("searchbox", { name: "Search workflow families" }),
    { target: { value: "Previous checkpoint" } });
  expect(screen.getByRole("heading", { name: "Family a" })).toBeInTheDocument();
  current = family("a", { dependency_count: 1, names: ["Replacement checkpoint"] });
  await client.invalidateQueries({ queryKey: ["workflow-families"] });
  await waitFor(() => expect(screen.queryByRole("heading", { name: "Family a" })).not.toBeInTheDocument());
  expect(screen.getByText("No workflows match these filters.")).toBeInTheDocument();
  fireEvent.change(screen.getByRole("searchbox", { name: "Search workflow families" }),
    { target: { value: "Replacement checkpoint" } });
  expect(screen.getByRole("heading", { name: "Family a" })).toBeInTheDocument();
});
