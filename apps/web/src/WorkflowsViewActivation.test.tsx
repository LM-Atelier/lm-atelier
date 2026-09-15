import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import type { Workflow, WorkflowActivation, WorkflowActivationChoice, WorkflowActivationPreparation } from "./types";

vi.mock("./api", () => ({ api: {
  workflowSummaries: vi.fn(), workflow: vi.fn(), workflowFamilies: vi.fn(),
  prepareWorkflowActivation: vi.fn(), activateWorkflowRevision: vi.fn(),
} }));
vi.mock("./CustomNodesPanel", () => ({ CustomNodesPanel: () => null }));
vi.mock("./RegistryInstallsPanel", () => ({ RegistryInstallsPanel: () => null }));
vi.mock("./WorkflowRevisionReviewPanel", () => ({ WorkflowRevisionReviewPanel: () => null }));

const digest = "b".repeat(64);
const clients: QueryClient[] = [];
const detail = (): Workflow => ({
  id: "workflow-a", family_id: null, name: "Neutral workflow", description: "",
  operation: "text_to_image", current_revision_id: "revision-a",
  revisions: ["older", "revision-a"].map((id, index) => ({
    id, workflow_id: "workflow-a", version: index + 1, engine: "comfyui",
    engine_version: null, trusted: true, created_at: "2026-09-01T00:00:00Z",
    dependency_contract_sha256: index === 1 ? digest : null,
    ui_graph_json: {}, api_graph_json: {}, input_schema_json: {},
    dependencies_json: { version: 1, slots: [] },
  })),
});
function choice(name: string, key = "default", slot = "style"): WorkflowActivationChoice {
  return { name, selection: {
    slot_name: slot, requirement_key: key, local_kind: "model_asset", local_id: name,
    recorded_resource_identity_sha256: "c".repeat(64), mount: {},
  } };
}
function prepared(overrides: Partial<WorkflowActivationPreparation> = {}): WorkflowActivationPreparation {
  const selected = choice("First");
  return {
    workflow_revision_id: "revision-a", workflow_artifact_sha256: "a".repeat(64),
    dependency_contract_sha256: digest, state: "prepared", selections: [selected.selection],
    slots: [{ name: "style", resource_kind: "model_asset", required: true, satisfaction: "all_of",
      requirement_keys: ["default"], choices: [selected] }], issues: [], ...overrides,
  };
}
function ambiguous(): WorkflowActivationPreparation {
  const result = prepared();
  result.state = "needs_attention"; result.selections = null;
  result.slots[0].choices.push(choice("Second"));
  result.issues = [{ code: "ambiguous_dependency_binding", slot_name: "style" }];
  return result;
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.workflowSummaries).mockResolvedValue([{
    id: "workflow-a", family_id: null, name: "Neutral workflow", description: "",
    operation: "text_to_image", current_revision_id: "revision-a", revision_count: 2,
    created_at: "2026-09-01T00:00:00Z", updated_at: "2026-09-01T00:00:00Z",
  }]);
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.workflow).mockImplementation(async () => detail());
  vi.mocked(api.prepareWorkflowActivation).mockResolvedValue(prepared());
  vi.mocked(api.activateWorkflowRevision).mockResolvedValue({
    id: "activation-a", workflow_revision_id: "revision-a", dependency_contract_sha256: digest,
    binding_sha256: "d".repeat(64), launch_sha256: "e".repeat(64), state: "ready", is_active: true,
  });
});
afterEach(() => { cleanup(); clients.splice(0).forEach(client => client.clear()); });
async function open() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><WorkflowsView /></QueryClientProvider>);
  fireEvent.click(await screen.findByText("Neutral workflow"));
  await screen.findByRole("button", { name: "New revision" });
  return client;
}
const activate = () => screen.getByRole("button", { name: "Activate dependencies" });
async function choose() {
  fireEvent.click(screen.getByRole("button", { name: "Choose dependencies" }));
  await screen.findByRole("button", { name: "Refresh dependencies" });
}
function selectNamed(label: string, name: string) {
  const control = screen.getByRole("combobox", { name: label });
  const option = within(control).getByRole("option", { name }) as HTMLOptionElement;
  fireEvent.change(control, { target: { value: option.value } });
}

it("activates a reviewed current revision with one click and refreshes readiness consumers", async () => {
  const client = await open();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  expect(api.prepareWorkflowActivation).not.toHaveBeenCalled();
  fireEvent.click(activate());
  await screen.findByText("Dependencies activated.");
  expect(api.prepareWorkflowActivation).toHaveBeenCalledExactlyOnceWith("workflow-a", "revision-a", expect.any(AbortSignal));
  expect(api.activateWorkflowRevision).toHaveBeenCalledExactlyOnceWith("workflow-a", "revision-a", {
    workflow_artifact_sha256: "a".repeat(64), dependency_contract_sha256: digest,
    selections: [choice("First").selection],
  });
  for (const key of ["workflows", "workflow-families", "workflow-family", "studio-capabilities"]) {
    expect(invalidate).toHaveBeenCalledWith({ queryKey: [key] });
  }
  expect(screen.queryByText(digest)).not.toBeInTheDocument();
});

it("never duplicates preparation or activation during a pending click", async () => {
  let finish!: (value: WorkflowActivationPreparation) => void;
  vi.mocked(api.prepareWorkflowActivation).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  await open();
  const button = activate(); fireEvent.click(button); fireEvent.click(button);
  expect(api.prepareWorkflowActivation).toHaveBeenCalledTimes(1);
  await act(async () => finish(prepared()));
  await screen.findByText("Dependencies activated.");
  expect(api.activateWorkflowRevision).toHaveBeenCalledTimes(1);
});

it("presents ambiguous matches by name and activates only the explicit choice", async () => {
  vi.mocked(api.prepareWorkflowActivation).mockResolvedValue(ambiguous());
  await open(); fireEvent.click(activate());
  await screen.findByRole("combobox", { name: "style" });
  expect(api.activateWorkflowRevision).not.toHaveBeenCalled();
  expect(activate()).toBeDisabled();
  selectNamed("style", "Second");
  fireEvent.click(activate());
  await screen.findByText("Dependencies activated.");
  expect(vi.mocked(api.activateWorkflowRevision).mock.calls[0][2].selections).toEqual([choice("Second").selection]);
  expect(api.prepareWorkflowActivation).toHaveBeenCalledTimes(2);
});

it("cannot activate a missing required resource and refresh never grants activation", async () => {
  const missing = ambiguous(); missing.slots[0].choices = [];
  missing.issues = [{ code: "missing_required_dependency", slot_name: "style" }];
  vi.mocked(api.prepareWorkflowActivation).mockResolvedValue(missing);
  await open(); fireEvent.click(activate());
  await screen.findByText("No matching installed resource. Install the required dependency, then refresh.");
  expect(activate()).toBeDisabled();
  vi.mocked(api.prepareWorkflowActivation).mockResolvedValue(prepared());
  fireEvent.click(screen.getByRole("button", { name: "Refresh dependencies" }));
  await waitFor(() => expect(activate()).toBeEnabled());
  expect(api.activateWorkflowRevision).not.toHaveBeenCalled();
  fireEvent.click(activate());
  await screen.findByText("Dependencies activated.");
});

it("does not enable optional resources implicitly", async () => {
  const optional = prepared(); optional.slots[0].required = false; optional.selections = [];
  vi.mocked(api.prepareWorkflowActivation).mockResolvedValue(optional);
  await open(); fireEvent.click(activate());
  await screen.findByText("Dependencies activated.");
  expect(vi.mocked(api.activateWorkflowRevision).mock.calls[0][2].selections).toEqual([]);
});

it("can explicitly enable an optional resource after inspecting choices", async () => {
  const optional = prepared(); optional.slots[0].required = false; optional.selections = [];
  vi.mocked(api.prepareWorkflowActivation).mockResolvedValue(optional);
  await open(); await choose();
  expect(api.activateWorkflowRevision).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("checkbox", { name: "Use style" }));
  fireEvent.click(activate());
  await screen.findByText("Dependencies activated.");
  expect(vi.mocked(api.activateWorkflowRevision).mock.calls[0][2].selections).toEqual([choice("First").selection]);
});

it("requires every member of an enabled optional all-of slot", async () => {
  const optional = prepared(); optional.slots[0].required = false; optional.selections = [];
  optional.slots[0].requirement_keys = ["default", "missing"];
  vi.mocked(api.prepareWorkflowActivation).mockResolvedValue(optional);
  await open(); await choose();
  fireEvent.click(screen.getByRole("checkbox", { name: "Use style" }));
  expect(activate()).toBeDisabled();
  expect(api.activateWorkflowRevision).not.toHaveBeenCalled();
});

it("keeps all-of requirements separate and any-of alternatives exclusive", async () => {
  const result = ambiguous();
  result.slots[0].satisfaction = "any_of";
  result.slots[0].requirement_keys = ["one", "two"];
  result.slots[0].choices = [choice("First", "one"), choice("Second", "two")];
  vi.mocked(api.prepareWorkflowActivation).mockResolvedValue(result);
  await open(); await choose();
  selectNamed("style", "Second");
  fireEvent.click(activate());
  await screen.findByText("Dependencies activated.");
  expect(vi.mocked(api.activateWorkflowRevision).mock.calls[0][2].selections).toEqual([choice("Second", "two").selection]);
});

it.each([
  { workflow_revision_id: "another-revision" },
  { dependency_contract_sha256: "f".repeat(64) },
  { workflow_artifact_sha256: "" },
])("refuses a mismatched or incomplete preparation %j", async (overrides) => {
  vi.mocked(api.prepareWorkflowActivation).mockResolvedValue(prepared(overrides));
  await open(); fireEvent.click(activate());
  await screen.findByText("The workflow changed. Refresh its details before activating dependencies.");
  expect(api.activateWorkflowRevision).not.toHaveBeenCalled();
});

it("does not silently replace an inspected resource when fresh preparation changes its identity", async () => {
  vi.mocked(api.prepareWorkflowActivation).mockResolvedValue(ambiguous());
  await open(); await choose(); selectNamed("style", "Second");
  const fresh = ambiguous();
  fresh.slots[0].choices[1].selection.recorded_resource_identity_sha256 = "f".repeat(64);
  vi.mocked(api.prepareWorkflowActivation).mockResolvedValue(fresh);
  fireEvent.click(activate());
  await screen.findByText("Dependencies changed. Check the choices and try again.");
  expect(api.activateWorkflowRevision).not.toHaveBeenCalled();
});

it("does not submit an old snapshot when the fresh preparation fails", async () => {
  await open(); await choose();
  vi.mocked(api.prepareWorkflowActivation).mockRejectedValue(new Error("Connection unavailable"));
  fireEvent.click(activate());
  await screen.findByText("Connection unavailable");
  expect(api.activateWorkflowRevision).not.toHaveBeenCalled();
});

it("abandons pending preparation after selecting another revision", async () => {
  let finish!: (value: WorkflowActivationPreparation) => void;
  vi.mocked(api.prepareWorkflowActivation).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  await open(); fireEvent.click(activate());
  const signal = vi.mocked(api.prepareWorkflowActivation).mock.calls[0][2];
  fireEvent.change(screen.getByRole("combobox", { name: "Revision" }), { target: { value: "older" } });
  expect(signal?.aborted).toBe(true);
  await act(async () => finish(prepared()));
  expect(api.activateWorkflowRevision).not.toHaveBeenCalled();
});

it("shows activation failure and never reports success from preparation alone", async () => {
  vi.mocked(api.activateWorkflowRevision).mockRejectedValue(Object.assign(new Error("Changed"), { code: "workflow-activation-unavailable" }));
  await open(); fireEvent.click(activate());
  await screen.findByText("The workflow or its dependencies changed. Refresh the choices and review the workflow again.");
  expect(screen.queryByText("Dependencies activated.")).not.toBeInTheDocument();
});

it.each(["legacy", "untrusted", "historical"])("does not offer activation for %s revisions", async (state) => {
  const value = detail();
  if (state === "legacy") value.revisions[1].dependency_contract_sha256 = null;
  if (state === "untrusted") value.revisions[1].trusted = false;
  vi.mocked(api.workflow).mockResolvedValue(value);
  await open();
  if (state === "historical") fireEvent.change(screen.getByRole("combobox", { name: "Revision" }), { target: { value: "older" } });
  expect(screen.queryByRole("button", { name: "Activate dependencies" })).not.toBeInTheDocument();
  expect(api.prepareWorkflowActivation).not.toHaveBeenCalled();
});


it("refreshes readiness after leaving a revision while activation is submitted", async () => {
  let finish!: (value: WorkflowActivation) => void;
  vi.mocked(api.activateWorkflowRevision).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  const client = await open();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  fireEvent.click(activate());
  await waitFor(() => expect(api.activateWorkflowRevision).toHaveBeenCalledTimes(1));
  const signal = vi.mocked(api.prepareWorkflowActivation).mock.calls[0][2];
  expect(vi.mocked(api.activateWorkflowRevision).mock.calls[0][3]).toBeUndefined();
  fireEvent.change(screen.getByRole("combobox", { name: "Revision" }), { target: { value: "older" } });
  expect(signal?.aborted).toBe(true);
  await act(async () => finish({
    id: "activation-a", workflow_revision_id: "revision-a", dependency_contract_sha256: digest,
    binding_sha256: "d".repeat(64), launch_sha256: "e".repeat(64), state: "ready", is_active: true,
  }));
  await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: ["workflow-families"] }));
  expect(screen.queryByText("Dependencies activated.")).not.toBeInTheDocument();
});

it("does not report success for another revision's activation response", async () => {
  vi.mocked(api.activateWorkflowRevision).mockResolvedValue({
    id: "activation-a", workflow_revision_id: "another-revision", dependency_contract_sha256: digest,
    binding_sha256: "d".repeat(64), launch_sha256: "e".repeat(64), state: "ready", is_active: true,
  });
  await open(); fireEvent.click(activate());
  await screen.findByText("Activation could not be confirmed. Refresh the workflow details.");
  expect(screen.queryByText("Dependencies activated.")).not.toBeInTheDocument();
});
