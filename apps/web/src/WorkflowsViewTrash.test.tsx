import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { api } from "./api";
import { WorkflowsView } from "./WorkflowsView";
import { mockWorkflowFamilyPages } from "./workflowFamilyReadFixtures";
import { mockWorkflowReadsFromFixture } from "./workflowReadFixtures";
import { recoveryImpact, recoveryItem } from "./test/recoveryFixtures";
import type { Workflow, WorkflowFamily } from "./types";

vi.mock("./api", async (original) => ({ ...(await original<typeof import("./api")>()), api: {
  workflows: vi.fn(), workflowSummaries: vi.fn(), workflow: vi.fn(), workflowFamilies: vi.fn(),
  workflowFamilyOperations: vi.fn(), setWorkflowFamilyPreference: vi.fn(), updateWorkflowFamily: vi.fn(),
  workflowDeletionImpact: vi.fn(), trashWorkflow: vi.fn(), recoveryImpact: vi.fn(), restoreRecovery: vi.fn(),
} }));
vi.mock("./CustomNodesPanel", () => ({ CustomNodesPanel: () => null }));
vi.mock("./RegistryInstallsPanel", () => ({ RegistryInstallsPanel: () => null }));
vi.mock("./WorkflowRevisionReviewPanel", () => ({ WorkflowRevisionReviewPanel: () => null }));
vi.mock("./WorkflowDiscover", () => ({ WorkflowDiscover: () => <p>Discover workflows</p> }));

const workflow: Workflow = { id: "garden", name: "Garden workflow", description: "Neutral fixture", operation: "text_to_image",
  current_revision_id: "revision-garden", revisions: [{ id: "revision-garden", workflow_id: "garden", version: 1,
    engine: "comfyui", engine_version: null, ui_graph_json: {}, api_graph_json: {}, input_schema_json: {},
    dependencies_json: {}, trusted: false, created_at: "2026-09-08T00:00:00Z" }] };
const family: WorkflowFamily = { id: "family-garden", name: "Garden", description: "", use_case: "", tags: [],
  enabled: true, archived: false, compatibility: false, created_at: "2026-09-08T00:00:00Z", updated_at: "2026-09-08T00:00:00Z",
  preferences: [], variants: [{ id: "garden", variant_key: "image", name: "Garden workflow", operation: "text_to_image",
    current_revision_id: "revision-garden", current_revision_version: 1, engine: "comfyui", capabilities: ["image"],
    trusted: false, readiness: "review_required", readiness_reason: "revision_untrusted" }] };
let client: QueryClient;
beforeEach(() => {
  vi.resetAllMocks();
  mockWorkflowReadsFromFixture(() => api.workflows());
  vi.mocked(api.workflows).mockResolvedValue([workflow]);
  mockWorkflowFamilyPages([family]);
  vi.mocked(api.workflowDeletionImpact).mockResolvedValue({ ...recoveryImpact(family.id), kind: "workflow_family" });
  vi.mocked(api.trashWorkflow).mockImplementation(async () => {
    vi.mocked(api.workflows).mockResolvedValue([]); mockWorkflowFamilyPages([]);
    return { ...recoveryItem(family.id), kind: "workflow_family", display_label: family.name };
  });
  vi.mocked(api.recoveryImpact).mockResolvedValue({ ...recoveryImpact(family.id, ["restore"]), kind: "workflow_family" });
  vi.mocked(api.restoreRecovery).mockResolvedValue({ kind: "workflow_family", subject_id: family.id,
    deletion_id: `deleted-${family.id}`, action: "restore", replayed: false, reclaimed_bytes: 0 });
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
});
afterEach(() => { cleanup(); client.clear(); });

it("keeps immediate Undo available after deleting the selected workflow and preserves loaded history", async () => {
  const history = { messages: ["Neutral historical result"] };
  client.setQueryData(["chat", "loaded-chat"], history);
  const invalidate = vi.spyOn(client, "invalidateQueries");
  render(<QueryClientProvider client={client}><WorkflowsView /></QueryClientProvider>);
  fireEvent.click(await screen.findByText(workflow.name));
  fireEvent.click(await screen.findByRole("button", { name: "Delete workflow family" }));
  const dialog = await screen.findByRole("dialog", { name: "Move this workflow family to Recently Deleted?" });
  fireEvent.click(await within(dialog).findByRole("button", { name: "Move to Recently Deleted" }));
  await waitFor(() => expect(api.trashWorkflow).toHaveBeenCalledOnce());
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  await waitFor(() => expect(screen.queryByRole("button", { name: "New revision" })).not.toBeInTheDocument());
  const undo = await screen.findByRole("button", { name: "Undo" });
  expect(undo).toHaveFocus();
  expect(client.getQueryData(["chat", "loaded-chat"])).toBe(history);
  expect(invalidate).not.toHaveBeenCalledWith({ queryKey: ["chat"] });
  for (const key of ["workflows", "workflow-families", "workflow-revision", "recovery-items"])
    expect(invalidate).toHaveBeenCalledWith({ queryKey: [key] });
  fireEvent.click(screen.getByRole("button", { name: "Discover" }));
  expect(screen.getByRole("button", { name: "Undo" })).toBeVisible();
  fireEvent.click(undo);
  await waitFor(() => expect(api.restoreRecovery).toHaveBeenCalledOnce());
  expect(api.updateWorkflowFamily).not.toHaveBeenCalled();
  expect(api.setWorkflowFamilyPreference).not.toHaveBeenCalled();
});
