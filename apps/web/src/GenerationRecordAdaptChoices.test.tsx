import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { GenerationRecordAdapt } from "./GenerationRecordAdapt";
import { PictureRemixDialog } from "./PictureRemixDialog";
import type { ModelProfile, WorkflowSummary } from "./types";

vi.mock("./api", async importOriginal => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, profiles: vi.fn(), profilesPage: vi.fn(),
    workflowSummaries: vi.fn(), workflowRevisionChoices: vi.fn(), workflowReadyRevisions: vi.fn(), remixPreview: vi.fn(),
    createChat: vi.fn(), adaptGenerationRecord: vi.fn(), remixPicture: vi.fn() } };
});

const content = new Uint8Array([1, 2, 3]).buffer;
const models = Array.from({ length: 51 }, (_, index) => ({
  id: `model_${index + 1}`, name: `Picture model ${index + 1}`, role: "image",
  model_install_id: null, engine: "mock", use_case: "", load_settings_json: {},
  request_settings_json: {}, is_default: false,
} satisfies ModelProfile));
const workflows = Array.from({ length: 201 }, (_, index) => ({
  id: `workflow_${index + 1}`, name: `Picture workflow ${index + 1}`, operation: "text_to_image",
  current_revision_id: `revision_${index + 1}`, description: "", revision_count: 1,
  created_at: "2026-01-01", updated_at: "2026-01-01",
} satisfies WorkflowSummary));

beforeEach(() => {
  vi.mocked(api.profiles).mockResolvedValue(models);
  vi.mocked(api.profilesPage).mockImplementation(async options => {
    const rows = models.filter(row => row.role === options.role
      && (!options.profileIds || options.profileIds.includes(row.id))
      && (!options.search || row.name.toLowerCase().includes(options.search.toLowerCase())));
    return rows.slice(options.offset ?? 0, (options.offset ?? 0) + options.limit);
  });
  vi.mocked(api.workflowSummaries).mockImplementation(async (options = {}) => {
    const rows = workflows.filter(row => (!options.operation || row.operation === options.operation)
      && (!options.workflowIds || options.workflowIds.includes(row.id))
      && (!options.search || row.name.toLowerCase().includes(options.search.toLowerCase())));
    return rows.slice(options.offset ?? 0, (options.offset ?? 0) + (options.limit ?? rows.length));
  });
  vi.mocked(api.workflowReadyRevisions).mockResolvedValue([]);
  vi.mocked(api.workflowRevisionChoices).mockImplementation(async (_, options = {}) => workflows
    .filter(row => options.revisionIds?.includes(row.current_revision_id)).map(row => ({
      revision_id: row.current_revision_id, workflow_id: row.id, workflow_name: row.name,
      operation: row.operation, version: 1,
    })));
  vi.mocked(api.createChat).mockResolvedValue({ id: "chat_new" } as never);
  vi.mocked(api.adaptGenerationRecord).mockResolvedValue({});
});
afterEach(() => { cleanup(); vi.resetAllMocks(); });

function show(kind: "adapt-model" | "adapt-workflow" | "remix") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>
    {kind === "remix" ? <PictureRemixDialog artifactId={`sha256:${"a".repeat(64)}`} onClose={() => {}} />
      : <GenerationRecordAdapt content={content} requirements={[]} bundledInputs={[]} onStarted={() => {}}
        plan={{ digest: `sha256:${"b".repeat(64)}`, operation: "text_to_image", ready: false,
          refusals: [{ code: kind === "adapt-model" ? "replay-model-missing" : "replay-workflow-missing",
            sha256: null, reasons: [] }] }} />}
  </QueryClientProvider>);
}

it.each(["adapt-model", "remix"] as const)("pages model choices in %s and retains a selected model during search", async kind => {
  show(kind);
  await screen.findByRole("option", { name: "Picture model 1" });
  expect(screen.queryByRole("option", { name: "Picture model 51" })).toBeNull();
  fireEvent.click(await screen.findByRole("button", { name: "More models" }));
  await screen.findByRole("option", { name: "Picture model 51" });
  const model = screen.getByRole("combobox", { name: "Model" });
  fireEvent.change(model, { target: { value: "model_51" } });
  await waitFor(() => expect(api.profilesPage).toHaveBeenCalledWith(
    expect.objectContaining({ profileIds: ["model_51"], limit: 1, role: "image" }),
  ));
  fireEvent.change(screen.getByRole("textbox", { name: "Search models" }), { target: { value: "model 2" } });
  await waitFor(() => expect(api.profilesPage).toHaveBeenCalledWith(
    expect.objectContaining({ search: "model 2", limit: 50, offset: 0, role: "image" }),
  ));
  await waitFor(() => expect(screen.queryByRole("option", { name: "Picture model 1" })).toBeNull());
  expect(model).toHaveValue("model_51");
  expect(screen.getByRole("option", { name: "Picture model 51" })).toBeInTheDocument();
  expect(api.profiles).not.toHaveBeenCalled();
});

it.each(["adapt-model", "remix"] as const)("retries a later model page in %s without losing the first page", async kind => {
  const read = vi.mocked(api.profilesPage).getMockImplementation()!;
  let refused = false;
  vi.mocked(api.profilesPage).mockImplementation(async options => {
    if (options.offset === 50 && !refused) { refused = true; throw new Error("Choices unavailable."); }
    return read(options);
  });
  show(kind);
  fireEvent.click(await screen.findByRole("button", { name: "More models" }));
  await screen.findByRole("alert");
  expect(screen.getByRole("option", { name: "Picture model 1" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry model choices" }));
  await screen.findByRole("option", { name: "Picture model 51" });
  expect(vi.mocked(api.profilesPage).mock.calls.filter(([options]) => options.offset === 50)).toHaveLength(2);
  expect(api.profiles).not.toHaveBeenCalled();
});

it("pages replacement workflows and searches beyond the old 200-choice ceiling", async () => {
  show("adapt-workflow");
  const workflow = screen.getByRole("combobox", { name: "Workflow" });
  fireEvent.click(workflow);
  await screen.findByRole("option", { name: "Picture workflow 1" });
  expect(screen.queryByRole("option", { name: "Picture workflow 51" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Load more workflows" }));
  await screen.findByRole("option", { name: "Picture workflow 51" });
  fireEvent.change(workflow, { target: { value: "workflow 201" } });
  fireEvent.click(await screen.findByRole("option", { name: "Picture workflow 201" }));
  const make = screen.getByRole("button", { name: "Make a new version" });
  await waitFor(() => expect(make).toHaveAttribute("aria-disabled", "false"));
  expect(workflow).toHaveValue("Picture workflow 201");
  fireEvent.click(workflow);
  fireEvent.change(workflow, { target: { value: "workflow 2" } });
  await screen.findByRole("option", { name: "Picture workflow 2" });
  fireEvent.keyDown(workflow, { key: "Escape" });
  expect(workflow).toHaveValue("Picture workflow 201");
  fireEvent.click(make);
  await waitFor(() => expect(api.adaptGenerationRecord).toHaveBeenCalledExactlyOnceWith("chat_new", content,
    { workflowRevisionId: "revision_201", profileId: undefined, loras: [], inputs: [] }));
  expect(api.workflowRevisionChoices).toHaveBeenCalledWith(
    expect.any(AbortSignal), { revisionIds: ["revision_201"], role: "image", limit: 1 },
  );
  expect(vi.mocked(api.workflowSummaries).mock.calls.every(([options]) => options!.limit! <= 50)).toBe(true);
});

it("retries a later workflow page and makes the exact replacement chosen from it", async () => {
  const read = vi.mocked(api.workflowSummaries).getMockImplementation()!;
  let refused = false;
  vi.mocked(api.workflowSummaries).mockImplementation(async (options = {}, signal) => {
    if (options.offset === 50 && !refused) { refused = true; throw new Error("Choices unavailable."); }
    return read(options, signal);
  });
  show("adapt-workflow");
  fireEvent.click(screen.getByRole("combobox", { name: "Workflow" }));
  fireEvent.click(await screen.findByRole("button", { name: "Load more workflows" }));
  await screen.findByRole("alert");
  expect(screen.getByRole("option", { name: "Picture workflow 1" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry workflows" }));
  fireEvent.click(await screen.findByRole("option", { name: "Picture workflow 51" }));
  const make = screen.getByRole("button", { name: "Make a new version" });
  await waitFor(() => expect(make).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(make);
  await waitFor(() => expect(api.adaptGenerationRecord).toHaveBeenCalledExactlyOnceWith("chat_new", content,
    { workflowRevisionId: "revision_51", profileId: undefined, loras: [], inputs: [] }));
});
