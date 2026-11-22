import { mockWorkflowReadsFromFixture } from "./workflowReadFixtures";
import { afterEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import { openWorkflowEditorPopup, runWorkflowEditor } from "./workflowEditorBridge";

/** A returned edit held for a retry keeps its editor session reported while a retry could validate it.
 *
 * The server counts an open editor session as use of the media worker only
 * while the page holding it reports in, and the editor run stops reporting
 * when it ends. A returned edit whose validation failed stays on this page,
 * and retrying it needs the same media launch, so the page reports the
 * session until the retry succeeds, the edit is discarded or the view holding
 * it goes away. A refusal that a retry would meet again, such as a workflow
 * that cannot be compiled, stops the reports: the launch would otherwise be
 * held for an edit that can never be saved.
 */

vi.mock("./api", () => ({
  api: {
    workflows: vi.fn(), workflowSummaries: vi.fn(), workflow: vi.fn(),
    workflowFamilyOperations: vi.fn().mockResolvedValue([]), workflowFamilies: vi.fn().mockResolvedValue([]),
    createWorkflowEditorDraft: vi.fn(),
    consumeWorkflowEditor: vi.fn(),
    cancelWorkflowEditor: vi.fn().mockResolvedValue(undefined),
    heartbeatWorkflowEditor: vi.fn().mockResolvedValue(undefined),
    registryInstalls: vi.fn().mockResolvedValue([]),
    customNodes: vi.fn().mockResolvedValue([]),
  },
}));

// Only the editor run is replaced, so the page reports at the bridge's own pace.
vi.mock("./workflowEditorBridge", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./workflowEditorBridge")>()),
  openWorkflowEditorPopup: vi.fn(),
  runWorkflowEditor: vi.fn(),
}));

const REPORT_MS = 30_000;
const SESSION = ["wf-a", "session-1", "nonce-1"];
const INVENTORY_UNAVAILABLE = "The media runtime could not describe its workflow nodes.";
const CANNOT_COMPILE = "The returned visual workflow cannot be compiled.";
const RETRY = "Retry validating returned edit";

function workflow(id: string, name: string) {
  return {
    id, name, description: "", operation: "text_to_image", current_revision_id: id,
    revisions: [{
      id, workflow_id: id, version: 1, engine: "comfyui", api_graph_json: {}, ui_graph_json: {},
      input_schema_json: {}, dependencies_json: {}, trusted: true, created_at: "2026-08-06T00:00:00Z",
    }],
  };
}

async function advance(ms: number) {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms); });
}

/** A report the server refuses. A mock's spy attaches handlers to any promise
 * it returns, which marks a refusal nobody handles as handled, so this one
 * comes back as a bare thenable: left unhandled, it fails the run as an
 * unhandled rejection. */
function refusedReport(): Promise<void> {
  const refused = Promise.reject(new Error("neutral report refused"));
  return { then: refused.then.bind(refused), catch: refused.catch.bind(refused) } as unknown as Promise<void>;
}

/** A request the server answered with `status`, shaped as the client reports one. */
function refusal(status: number, message: string): Error {
  return Object.assign(new Error(message), { status });
}

/** A failed validation as the editor run passes it on while the ComfyUI window stays open. */
function keptOpen(error: Error): Error {
  return new Error(
    `${error.message} The ComfyUI window remains open so you can correct or download the workflow.`,
    { cause: error },
  );
}

/** Runs the editor until its returned edit fails validation with `failure`, by default one a retry can fix. */
async function holdReturnedEdit(failure: Error = keptOpen(new Error(INVENTORY_UNAVAILABLE))) {
  vi.useFakeTimers();
  vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
  vi.mocked(openWorkflowEditorPopup).mockReturnValue({ closed: false, close: vi.fn() } as unknown as Window);
  vi.mocked(runWorkflowEditor).mockImplementation(async (_client, _id, _popup, options) => {
    options?.onSubmission?.({
      workflowId: "wf-a", sessionId: "session-1", nonce: "nonce-1",
      baseRevisionId: "revision-1", uiGraph: { nodes: [1] }, apiPrompt: { 1: {} },
    });
    throw failure;
  });
  mockWorkflowReadsFromFixture(() => api.workflows());
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const view = render(
    <QueryClientProvider client={client}>
      <WorkflowsView />
    </QueryClientProvider>,
  );
  await advance(50);
  fireEvent.click(screen.getByText("Alpha"));
  await advance(50);
  fireEvent.click(screen.getByRole("button", { name: "Edit in ComfyUI (preview)" }));
  await advance(50);
  expect(screen.getByRole("button", { name: RETRY })).toBeEnabled();
  return view;
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.useRealTimers();
});

it("reports a held edit's session every 30 seconds for as long as its retry is offered", async () => {
  await holdReturnedEdit();
  vi.mocked(api.heartbeatWorkflowEditor).mockImplementationOnce(refusedReport);

  await advance(REPORT_MS);
  expect(api.heartbeatWorkflowEditor).toHaveBeenCalledExactlyOnceWith(...SESSION);
  // The server stops counting a session nobody has reported for three minutes.
  // A refused report is dropped quietly, and the next one goes out on time.
  await advance(6 * REPORT_MS);
  expect(vi.mocked(api.heartbeatWorkflowEditor).mock.calls).toEqual(Array(7).fill(SESSION));
  expect(screen.getByRole("button", { name: RETRY })).toBeEnabled();
});

it("goes on reporting after a failed retry and stops once a retry succeeds", async () => {
  await holdReturnedEdit();
  vi.mocked(api.consumeWorkflowEditor).mockRejectedValueOnce(new Error(INVENTORY_UNAVAILABLE));
  fireEvent.click(screen.getByRole("button", { name: RETRY }));
  await advance(50);
  expect(screen.getByRole("button", { name: RETRY })).toBeEnabled();

  await advance(REPORT_MS);
  expect(api.heartbeatWorkflowEditor).toHaveBeenCalledExactlyOnceWith(...SESSION);

  vi.mocked(api.consumeWorkflowEditor).mockResolvedValueOnce({ changed: false } as never);
  fireEvent.click(screen.getByRole("button", { name: RETRY }));
  await advance(50);
  expect(screen.getByText("No workflow changes to save.")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: RETRY })).toBeNull();

  await advance(4 * REPORT_MS);
  expect(api.consumeWorkflowEditor).toHaveBeenCalledTimes(2);
  expect(api.heartbeatWorkflowEditor).toHaveBeenCalledOnce();
});

it("stops reporting once the held edit is discarded", async () => {
  await holdReturnedEdit();
  await advance(REPORT_MS);
  expect(api.heartbeatWorkflowEditor).toHaveBeenCalledExactlyOnceWith(...SESSION);

  fireEvent.click(screen.getByRole("button", { name: "Discard pending edit" }));
  await advance(50);
  expect(api.cancelWorkflowEditor).toHaveBeenCalledExactlyOnceWith(...SESSION);
  expect(screen.queryByRole("button", { name: RETRY })).toBeNull();

  await advance(4 * REPORT_MS);
  expect(api.heartbeatWorkflowEditor).toHaveBeenCalledOnce();
});

it("stops reporting when the page holding the edit goes away", async () => {
  const view = await holdReturnedEdit();
  await advance(REPORT_MS);
  expect(api.heartbeatWorkflowEditor).toHaveBeenCalledExactlyOnceWith(...SESSION);

  view.unmount();
  await advance(4 * REPORT_MS);
  expect(api.heartbeatWorkflowEditor).toHaveBeenCalledOnce();
});

it("does not report a held edit whose validation was refused in a way no retry can change", async () => {
  await holdReturnedEdit(keptOpen(refusal(422, CANNOT_COMPILE)));

  await advance(4 * REPORT_MS);
  expect(api.heartbeatWorkflowEditor).not.toHaveBeenCalled();
  // The edit is still held, so it can be retried, downloaded or discarded.
  expect(screen.getByRole("button", { name: RETRY })).toBeEnabled();
});

it("reports a held edit only while its last failure is one a retry could fix", async () => {
  await holdReturnedEdit(keptOpen(refusal(503, INVENTORY_UNAVAILABLE)));
  await advance(REPORT_MS);
  expect(api.heartbeatWorkflowEditor).toHaveBeenCalledExactlyOnceWith(...SESSION);

  vi.mocked(api.consumeWorkflowEditor).mockRejectedValueOnce(refusal(422, CANNOT_COMPILE));
  fireEvent.click(screen.getByRole("button", { name: RETRY }));
  await advance(50);
  expect(screen.getByRole("button", { name: RETRY })).toBeEnabled();
  await advance(4 * REPORT_MS);
  expect(api.heartbeatWorkflowEditor).toHaveBeenCalledOnce();

  // A later retry the server could not validate is worth trying again.
  vi.mocked(api.consumeWorkflowEditor).mockRejectedValueOnce(refusal(503, INVENTORY_UNAVAILABLE));
  fireEvent.click(screen.getByRole("button", { name: RETRY }));
  await advance(50);
  await advance(REPORT_MS);
  expect(vi.mocked(api.heartbeatWorkflowEditor).mock.calls).toEqual(Array(2).fill(SESSION));
});
