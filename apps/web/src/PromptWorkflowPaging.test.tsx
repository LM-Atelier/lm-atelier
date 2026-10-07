import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { api } from "./api";
import { PromptTemplateImageSetupPicker } from "./PromptTemplateImageSetupPicker";
import { PromptLibraryView } from "./PromptLibraryView";
import type { PromptTemplateResourcePolicy, WorkflowReadyRevision } from "./types";

vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, workflowReadyRevisions: vi.fn(),
    workflowFamilies: vi.fn(), modelAssets: vi.fn(), promptTemplates: vi.fn(),
    promptTemplate: vi.fn(), promptTemplateRevisions: vi.fn() } };
});

const rows: WorkflowReadyRevision[] = Array.from({ length: 52 }, (_, index) => ({
  family_id: `family-${index}`, family_name: `Family ${index}`, workflow_id: `workflow-${index}`,
  workflow_name: `Variant ${index}`, revision_id: `revision-${index}`, revision_version: 1,
  operation: "text_to_image",
}));
const saved = rows[51];
const fixed = { mode: "fixed" as const, workflow_revision_id: saved.revision_id,
  lora_policy: { mode: "none" as const } };

beforeEach(() => {
  vi.mocked(api.modelAssets).mockResolvedValue([]);
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.workflowReadyRevisions).mockImplementation(async options => {
    if (options?.revisionIds) return rows.filter(row => options.revisionIds!.includes(row.revision_id));
    const found = rows.filter(row => !options?.search || row.family_name.includes(options.search));
    return found.slice(options?.offset ?? 0, (options?.offset ?? 0) + (options?.limit ?? 50));
  });
});
afterEach(() => { cleanup(); vi.resetAllMocks(); });

function mount(element: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>{element}</QueryClientProvider>);
  return client;
}

it("keeps the saved quick setup outside the first page and during search", async () => {
  const onChange = vi.fn();
  mount(<PromptTemplateImageSetupPicker value={fixed} onChange={onChange} />);
  const picker = screen.getByRole("combobox", { name: "Template workflow" });
  await within(picker).findByRole("option", { name: "Family 51 - Variant 51" });
  expect(picker).toHaveValue(saved.revision_id);
  fireEvent.change(screen.getByRole("searchbox"), { target: { value: "Family 50" } });
  await within(picker).findByRole("option", { name: "Family 50 - Variant 50" });
  expect(picker).toHaveValue(saved.revision_id);
  expect(within(picker).getByRole("option", { name: "Family 51 - Variant 51" })).toBeInTheDocument();
  expect(onChange).not.toHaveBeenCalled();
  expect(api.workflowReadyRevisions).toHaveBeenCalledWith(
    expect.objectContaining({ revisionIds: [saved.revision_id], limit: 200 }), expect.any(AbortSignal),
  );
});

it("retries a later page without replacing the saved quick setup", async () => {
  const normal = vi.mocked(api.workflowReadyRevisions).getMockImplementation()!;
  let fail = true;
  vi.mocked(api.workflowReadyRevisions).mockImplementation(async (options, signal) => {
    if (options?.offset === 50 && fail) throw new Error("Later page unavailable");
    return normal(options, signal);
  });
  const onChange = vi.fn();
  mount(<PromptTemplateImageSetupPicker value={fixed} onChange={onChange} />);
  fireEvent.click(await screen.findByRole("button", { name: "Load more ready image workflows" }));
  await screen.findByText("Later page unavailable");
  expect(screen.getByLabelText("Template workflow")).toHaveValue(saved.revision_id);
  fail = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry ready image workflows" }));
  await screen.findByRole("option", { name: "Family 50 - Variant 50" });
  expect(onChange).not.toHaveBeenCalled();
});

it("does not call a saved choice unavailable while its exact read is pending or failed", async () => {
  let reject!: (error: Error) => void;
  const exact = new Promise<WorkflowReadyRevision[]>((_, fail) => { reject = fail; });
  vi.mocked(api.workflowReadyRevisions).mockImplementation(async options => options?.revisionIds ? exact : [saved]);
  mount(<PromptTemplateImageSetupPicker value={fixed} onChange={vi.fn()} />);
  await screen.findByText("Checking selected workflows…");
  expect(screen.queryByText(/currently unavailable/)).not.toBeInTheDocument();
  reject(new Error("Exact read failed"));
  await screen.findByRole("button", { name: "Retry selected workflows" });
  expect(screen.getByLabelText("Template workflow")).toHaveValue(saved.revision_id);
  expect(screen.queryByText(/currently unavailable/)).not.toBeInTheDocument();
  expect(screen.queryByRole("option", { name: "Family 51 - Variant 51" })).not.toBeInTheDocument();
});

it("uses the exact missing result instead of reviving a saved choice from a stale page", async () => {
  vi.mocked(api.workflowReadyRevisions).mockImplementation(async options => options?.revisionIds ? [] : [saved]);
  mount(<PromptTemplateImageSetupPicker value={fixed} onChange={vi.fn()} />);
  await screen.findByRole("option", { name: "Previously selected workflow (currently unavailable)" });
  expect(screen.getByLabelText("Template workflow")).toHaveValue(saved.revision_id);
  expect(screen.queryByRole("option", { name: "Family 51 - Variant 51" })).not.toBeInTheDocument();
});

async function editLibrary(resources: PromptTemplateResourcePolicy) {
  const stamp = "2026-09-01T00:00:00Z";
  const definition = { id: "template", name: "Neutral template", description: "", archived: false,
    current_revision_id: "template-revision", created_at: stamp, updated_at: stamp };
  const revision = { id: "template-revision", prompt_template_id: definition.id, version: 1,
    schema_version: 1 as const, contract_sha256: "a".repeat(64), created_at: stamp,
    contract_json: { schema_version: 1 as const, operation: "text_to_image" as const,
      body: "A blue cube", slots: [], resource_policy: resources } };
  vi.mocked(api.promptTemplates).mockResolvedValue({ items: [definition], total: 1, offset: 0, limit: 50 });
  vi.mocked(api.promptTemplate).mockResolvedValue({ ...definition, current_revision: revision });
  vi.mocked(api.promptTemplateRevisions).mockResolvedValue([revision]);
  mount(<PromptLibraryView />);
  fireEvent.click(await screen.findByRole("button", { name: "Edit" }));
}

it("resolves the fixed library workflow independently of the visible page", async () => {
  await editLibrary(fixed);
  await screen.findByRole("option", { name: "Family 51 - Variant 51 - revision 1" });
  expect(screen.getByLabelText("Workflow")).toHaveValue(saved.revision_id);
  fireEvent.change(screen.getByRole("searchbox"), { target: { value: "Family 50" } });
  await screen.findByRole("option", { name: "Family 50 - Variant 50 - revision 1" });
  expect(screen.getByLabelText("Workflow")).toHaveValue(saved.revision_id);
});

it("looks up all pinned pool choices in one exact batch and preserves them on page failure", async () => {
  await editLibrary({ mode: "pool", strategy: "round_robin", options: [
    { workflow_revision_id: rows[50].revision_id, lora_policy: { mode: "none" } }, fixed,
  ] });
  await screen.findAllByRole("option", { name: "Family 51 · Variant 51 (ready)" });
  expect(screen.getByLabelText("Option 1 workflow revision")).toHaveValue(rows[50].revision_id);
  expect(screen.getByLabelText("Option 2 workflow revision")).toHaveValue(saved.revision_id);
  expect(api.workflowReadyRevisions).toHaveBeenCalledWith(
    expect.objectContaining({ revisionIds: [rows[50].revision_id, saved.revision_id] }), expect.any(AbortSignal),
  );
  vi.mocked(api.workflowReadyRevisions).mockRejectedValue(new Error("Search unavailable"));
  fireEvent.change(screen.getByRole("searchbox"), { target: { value: "Other" } });
  await waitFor(() => expect(screen.getByText("Search unavailable")).toBeInTheDocument());
  expect(screen.getByLabelText("Option 1 workflow revision")).toHaveValue(rows[50].revision_id);
  expect(screen.getByLabelText("Option 2 workflow revision")).toHaveValue(saved.revision_id);
});
