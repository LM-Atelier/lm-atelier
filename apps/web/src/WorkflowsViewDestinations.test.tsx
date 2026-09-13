import { mockWorkflowReadsFromFixture } from "./workflowReadFixtures";
import { afterEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import { openWorkflowEditorPopup, runWorkflowEditor } from "./workflowEditorBridge";

/** Moving between the library and Discover must not cost anybody their work.
 *
 * The pending-edit states below are the same ones the recovery tests beside
 * this file build, deliberately: the question here is not whether recovery
 * works, which those pin, but whether looking at Discover and coming back
 * leaves it exactly where it was.
 */

vi.mock("./api", () => ({
  api: {
    workflows: vi.fn(), workflowSummaries: vi.fn(), workflow: vi.fn(),
    workflowFamilies: vi.fn().mockResolvedValue([]),
    workflowCatalog: vi.fn().mockResolvedValue({ items: [], next_cursor: null, stale: false }),
    validateWorkflow: vi.fn(),
    createWorkflow: vi.fn(),
    createWorkflowEditorDraft: vi.fn(),
    consumeWorkflowEditor: vi.fn(),
    cancelWorkflowEditor: vi.fn().mockResolvedValue(undefined),
    workflowOpenTarget: vi.fn(),
    registryInstalls: vi.fn().mockResolvedValue([]),
    customNodes: vi.fn().mockResolvedValue([]),
  },
}));

vi.mock("./workflowEditorBridge", () => ({
  openWorkflowEditorPopup: vi.fn(),
  runWorkflowEditor: vi.fn(),
}));

function workflow(id: string, name: string) {
  return {
    id, name, description: "", operation: "text_to_image", current_revision_id: id,
    revisions: [{
      id, workflow_id: id, version: 1, engine: "comfyui", api_graph_json: {}, ui_graph_json: {},
      input_schema_json: {}, dependencies_json: {}, trusted: true, created_at: "2026-08-06T00:00:00Z",
    }],
  };
}

function editorReturn() {
  return {
    validated_return_id: "return-1", session_id: "session-1", workflow_id: "wf-a",
    base_revision_id: "revision-1", current_revision_id: "revision-1",
    base_graph_sha256: "a".repeat(64), returned_graph_sha256: "b".repeat(64),
    base_prompt_sha256: "c".repeat(64), returned_prompt_sha256: "d".repeat(64),
    changed: true, forked: false,
    delta: {
      node_count_delta: 0, link_count_delta: 0, added_node_types: [], removed_node_types: [],
      added_asset_filenames: [], removed_asset_filenames: [],
    },
    expires_at: "2026-08-08T18:00:00Z",
  };
}

function renderView() {
  mockWorkflowReadsFromFixture(() => api.workflows());
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <WorkflowsView />
    </QueryClientProvider>,
  );
  return client;
}

function openDestination(name: "Library" | "Discover") {
  fireEvent.click(screen.getByRole("button", { name }));
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("opens on the library and asks no remote source anything until Discover is chosen", async () => {
  vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
  renderView();
  await screen.findByText("Alpha");

  expect(screen.getByRole("button", { name: "Library" })).toHaveAttribute("aria-current", "page");
  expect(screen.getByRole("button", { name: "Discover" })).not.toHaveAttribute("aria-current");
  // Somebody who never opens Discover must not cause a request to a
  // rate-limited provider just by visiting their own library.
  expect(api.workflowCatalog).not.toHaveBeenCalled();

  openDestination("Discover");

  expect(await screen.findByRole("region", { name: "Discover workflows" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Discover" })).toHaveAttribute("aria-current", "page");
  await waitFor(() => expect(api.workflowCatalog).toHaveBeenCalledOnce());
});

it("keeps an unsaved validated edit through a visit to Discover", async () => {
  vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
  vi.mocked(openWorkflowEditorPopup).mockReturnValue({ closed: false, close: vi.fn() } as unknown as Window);
  vi.mocked(runWorkflowEditor).mockImplementation(async (_client, _id, _popup, options) => {
    options?.onValidated?.(editorReturn());
    throw new Error("neutral round trip interrupted");
  });
  vi.mocked(api.createWorkflowEditorDraft).mockResolvedValue(undefined as never);
  renderView();
  fireEvent.click(await screen.findByText("Alpha"));
  fireEvent.click(await screen.findByRole("button", { name: "Edit in ComfyUI (preview)" }));
  await screen.findByRole("button", { name: "Retry saving validated edit" });

  openDestination("Discover");
  await screen.findByRole("region", { name: "Discover workflows" });
  // Out of view. This pending edit lives on the page itself, so it would survive
  // an unmount too - the filter case below is the one that tells them apart.
  expect(screen.queryByRole("button", { name: "Retry saving validated edit" })).toBeNull();

  openDestination("Library");

  const retry = await screen.findByRole("button", { name: "Retry saving validated edit" });
  expect(retry).toBeEnabled();
  fireEvent.click(retry);
  // The retry still carries the identifiers of the edit made before the visit,
  // which is what proves the state survived rather than being rebuilt.
  await waitFor(() => expect(api.createWorkflowEditorDraft).toHaveBeenCalledWith("wf-a", "return-1"));
});

it("shows selected-detail recovery on return when the detail failed while away", async () => {
  vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
  const client = renderView();
  fireEvent.click(await screen.findByText("Alpha"));
  await screen.findByRole("button", { name: "New revision" });

  openDestination("Discover");
  vi.mocked(api.workflow).mockRejectedValue(new Error("neutral detail refresh failed"));
  await act(async () => { await client.invalidateQueries({ queryKey: ["workflows", "detail", "wf-a"] }); });

  openDestination("Library");

  expect(await screen.findByText("neutral detail refresh failed")).toBeTruthy();
  const retry = screen.getByRole("button", { name: "Retry workflow details" });
  vi.mocked(api.workflow).mockResolvedValue(workflow("wf-a", "Alpha") as never);
  fireEvent.click(retry);
  expect(await screen.findByRole("button", { name: "New revision" })).toBeTruthy();
  client.clear();
});

it("brings the library into view before opening something that lives there", async () => {
  // A dialog opened inside the hidden half would lock the page while showing
  // nothing, so a library action taken from Discover returns to the library.
  vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
  renderView();
  await screen.findByText("Alpha");
  openDestination("Discover");
  await screen.findByRole("region", { name: "Discover workflows" });

  fireEvent.click(screen.getByRole("button", { name: "New workflow" }));

  expect(screen.getByRole("button", { name: "Library" })).toHaveAttribute("aria-current", "page");
  expect(await screen.findByRole("dialog")).toBeTruthy();
});

it("keeps a Discover search in place when looking back at the library", async () => {
  vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
  renderView();
  await screen.findByText("Alpha");
  openDestination("Discover");
  fireEvent.change(await screen.findByLabelText("Search workflows"), { target: { value: "portrait" } });

  openDestination("Library");
  openDestination("Discover");

  expect((screen.getByLabelText("Search workflows") as HTMLInputElement).value).toBe("portrait");
});

it("keeps the library's own filters through a visit to Discover", async () => {
  // This is the case that actually distinguishes hiding from unmounting. The
  // pending-edit state above lives on the page itself and would survive either
  // way; the family list's search and filters live inside the library half and
  // would be thrown away by an unmount.
  vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
  renderView();
  await screen.findByText("Alpha");
  fireEvent.change(screen.getByLabelText("Search workflow families"), { target: { value: "alp" } });

  openDestination("Discover");
  await screen.findByRole("region", { name: "Discover workflows" });
  openDestination("Library");

  expect((screen.getByLabelText("Search workflow families") as HTMLInputElement).value).toBe("alp");
});
