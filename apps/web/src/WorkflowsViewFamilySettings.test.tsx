import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { api } from "./api";
import { WorkflowsView } from "./WorkflowsView";
import { mockWorkflowFamilyPages } from "./workflowFamilyReadFixtures";
import type { Workflow, WorkflowFamily } from "./types";

vi.mock("./api", () => ({ api: {
  workflowSummaries: vi.fn(), workflow: vi.fn(), workflowFamilies: vi.fn(), workflowFamilyOperations: vi.fn(),
  updateWorkflowFamily: vi.fn(), setWorkflowFamilyPreference: vi.fn(),
} }));
vi.mock("./CustomNodesPanel", () => ({ CustomNodesPanel: () => null }));
vi.mock("./RegistryInstallsPanel", () => ({ RegistryInstallsPanel: () => null }));
vi.mock("./WorkflowRevisionReviewPanel", () => ({ WorkflowRevisionReviewPanel: () => null }));
vi.mock("./WorkflowFamilyDependencies", () => ({ WorkflowFamilyDependencies: () => null }));
vi.mock("./WorkflowFamilyUsage", () => ({ WorkflowFamilyUsage: () => null }));

const stamp = "2026-09-08T00:00:00Z";
let family: WorkflowFamily;
const clients: QueryClient[] = [];

beforeEach(() => {
  vi.resetAllMocks();
  family = {
    id: "mixed-family", name: "Mixed family", description: "", use_case: "Neutral scenes", tags: [],
    enabled: true, archived: false, compatibility: false, created_at: stamp, updated_at: stamp,
    preferences: [
      { selector_capability: "image", enabled: true, is_default: false, sort_order: 0 },
      { selector_capability: "video", enabled: true, is_default: true, sort_order: 0 },
    ],
    variants: ["image", "video"].map(kind => ({
      id: `${kind}-variant`, variant_key: kind, name: `${kind} choice`,
      operation: kind === "image" ? "text_to_image" : "text_to_video",
      current_revision_id: `${kind}-revision`, current_revision_version: 1,
      engine: "mock", capabilities: [], trusted: true, readiness: "ready", readiness_reason: null,
    })),
  };
  mockWorkflowFamilyPages(async () => [family]);
  vi.mocked(api.workflowSummaries).mockResolvedValue([]);
  vi.mocked(api.workflow).mockImplementation(async id => {
    const variant = family.variants.find(row => row.id === id)!;
    return {
      id, family_id: family.id, name: variant.name, description: "", operation: variant.operation,
      current_revision_id: variant.current_revision_id, revisions: [{
        id: variant.current_revision_id!, workflow_id: id, version: 1, engine: "mock", engine_version: null,
        trusted: true, created_at: stamp, ui_graph_json: {}, api_graph_json: {},
        input_schema_json: {}, dependencies_json: {},
      }],
    } satisfies Workflow;
  });
  vi.mocked(api.updateWorkflowFamily).mockImplementation(async (_id, values) => {
    family = { ...family, ...values };
    return family;
  });
  vi.mocked(api.setWorkflowFamilyPreference).mockImplementation(async (_id, capability, values) => {
    const updated = { selector_capability: capability, ...values };
    family = { ...family, preferences: family.preferences.map(one =>
      one.selector_capability === capability ? updated : one) };
    return updated;
  });
});

afterEach(() => { cleanup(); clients.splice(0).forEach(client => client.clear()); });

async function open() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  client.setQueryData(["workflow-families", "library", false], [family]);
  render(<QueryClientProvider client={client}><WorkflowsView /></QueryClientProvider>);
  fireEvent.click(await screen.findByText("image choice"));
  await screen.findByRole("button", { name: "Edit family details" });
  return client;
}

it("saves family details while paged and selected-family caches are mounted", async () => {
  const client = await open();
  const operations = client.getQueryData(["workflow-families", "operations", false]);
  fireEvent.click(screen.getByRole("button", { name: "Edit family details" }));
  fireEvent.change(screen.getByRole("textbox", { name: "Family name" }), { target: { value: "Renamed family" } });
  fireEvent.click(screen.getByRole("button", { name: "Save family details" }));
  expect(await screen.findByText("Family details saved.")).toBeInTheDocument();
  expect(screen.queryByRole("textbox", { name: "Family name" })).not.toBeInTheDocument();
  expect(api.updateWorkflowFamily).toHaveBeenCalledWith("mixed-family", { name: "Renamed family" });
  expect(client.getQueryData<WorkflowFamily[]>(["workflow-families", "library", false])?.[0].name).toBe("Renamed family");
  expect(client.getQueryData(["workflow-families", "operations", false])).toEqual(operations);
  await waitFor(() => expect(within(screen.getByRole("region", { name: "Family details" }))
    .getByRole("heading", { name: "Renamed family" })).toBeInTheDocument());
});

it("offers and updates every family capability when one variant is selected", async () => {
  await open();
  expect(screen.getByRole("checkbox", { name: "Images" })).toBeChecked();
  const video = screen.getByRole("checkbox", { name: "Video Used when nobody chooses" });
  expect(video).toBeChecked();
  expect(screen.getByRole("button", { name: "Make this the default" })).toBeInTheDocument();
  fireEvent.click(video);
  await waitFor(() => expect(api.setWorkflowFamilyPreference).toHaveBeenCalledWith("mixed-family", "video",
    { enabled: false, is_default: false, sort_order: 0 }));
  await waitFor(() => expect(screen.getByRole("checkbox", { name: "Video" })).not.toBeChecked());
  expect(screen.getByRole("checkbox", { name: "Images" })).toBeChecked();
  fireEvent.click(screen.getByText("video choice"));
  await waitFor(() => expect(api.workflow).toHaveBeenCalledWith("video-variant", expect.any(AbortSignal)));
  await waitFor(() => expect(screen.getByRole("checkbox", { name: "Images" })).toBeChecked());
  await waitFor(() => expect(screen.getByRole("checkbox", { name: "Video" })).not.toBeChecked());
});
