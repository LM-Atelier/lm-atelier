import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { WorkflowRecipeEditor } from "./WorkflowRecipeEditor";
import type { EngineCapabilities, SettingField, WorkflowRevisionSchema, WorkflowSummary } from "./types";
import type { WorkflowUseCasePreset } from "./workflowUseCaseTypes";

vi.mock("./api", () => ({ api: { workflowSummaries: vi.fn(), engines: vi.fn(), workflowRevisionSchema: vi.fn() } }));
const clients: QueryClient[] = [];
const field: SettingField = { key: "steps", label: "Steps", type: "integer", default: 20, minimum: 1, maximum: 50,
  step: 1, choices: [], scope: "request", visibility: "basic", restart_required: false, available: true, unavailable_reason: null, help: "" };
const schema: WorkflowRevisionSchema = { workflow_id: "reference", revision_id: "revision", operation: "text_to_image",
  input_schema_json: { properties: { quality: { type: "number", default: 0.5, minimum: 0, maximum: 1 } } } };
const engine: EngineCapabilities = { engine: "mock", version: "1", roles: ["image"], operations: ["text_to_image"], formats: [], devices: [],
  streaming: false, tool_calling: false, settings: [field], healthy: true, details: {} };
const existing: WorkflowUseCasePreset = { id: "recipe", name: "Detailed", use_case: "image_generation", settings_json: { steps: 24, older_control: 0.7 }, enabled: true, is_default: false, builtin: false };
function show(recipe: WorkflowUseCasePreset | null = null) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  const save = vi.fn();
  render(<QueryClientProvider client={client}><WorkflowRecipeEditor recipe={recipe} saving={false} error={null} onSave={save} onCancel={vi.fn()} /></QueryClientProvider>);
  return { client, save };
}
async function reference() {
  await screen.findByRole("option", { name: "Reference workflow" });
  fireEvent.change(screen.getByRole("combobox", { name: "Reference workflow for settings" }), { target: { value: "reference" } });
  await screen.findByRole("checkbox", { name: "Include Steps" });
}
beforeEach(() => {
  vi.mocked(api.workflowSummaries).mockResolvedValue([{ id: "reference", name: "Reference workflow", operation: "text_to_image", description: "", current_revision_id: "revision", revision_count: 1, created_at: "", updated_at: "" }]);
  vi.mocked(api.engines).mockResolvedValue([engine]);
  vi.mocked(api.workflowRevisionSchema).mockResolvedValue(schema);
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); vi.resetAllMocks(); });

it("creates only explicitly included declared settings without binding a workflow", async () => {
  const { save } = show();
  fireEvent.change(screen.getByRole("textbox", { name: "Recipe name" }), { target: { value: "  Fine detail  " } });
  await reference();
  expect(screen.queryByRole("spinbutton", { name: "Steps" })).toBeNull();
  fireEvent.click(screen.getByRole("checkbox", { name: "Include Steps" }));
  fireEvent.change(screen.getByRole("spinbutton", { name: "Steps" }), { target: { value: "28" } });
  fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));
  expect(save).toHaveBeenCalledExactlyOnceWith({ name: "Fine detail", use_case: "image_generation", settings_json: { steps: 28 }, enabled: true, is_default: false });
});

it("preserves unknown saved settings until explicitly removed", async () => {
  const { save } = show(existing);
  await reference();
  expect(screen.getByText("older_control (not available in these controls)")).toBeVisible();
  fireEvent.change(screen.getByRole("spinbutton", { name: "Steps" }), { target: { value: "30" } });
  fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));
  expect(save.mock.calls[0][0].settings_json).toEqual({ steps: 30, older_control: 0.7 });
  fireEvent.click(screen.getByRole("button", { name: "Remove saved setting older_control" }));
  fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));
  expect(save.mock.calls[1][0].settings_json).toEqual({ steps: 30 });
  expect(existing.settings_json).toEqual({ steps: 24, older_control: 0.7 });
});

it("does not coerce fractional integers or save values outside declared bounds", async () => {
  const { save } = show({ ...existing, settings_json: { steps: 24 } });
  await reference();
  const input = screen.getByRole("spinbutton", { name: "Steps" });
  fireEvent.change(input, { target: { value: "3.5" } });
  fireEvent.blur(input);
  expect(input).toHaveValue(3.5);
  expect(screen.getByText("Choose a valid value for Steps.")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));
  expect(save).not.toHaveBeenCalled();
  fireEvent.change(input, { target: { value: "51" } });
  fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));
  expect(save).not.toHaveBeenCalled();
});

it("offers no prompt, read-only, load-time or unavailable setting controls", async () => {
  vi.mocked(api.engines).mockResolvedValue([{ ...engine, settings: [field,
    { ...field, key: "prompt", label: "Prompt" }, { ...field, key: "negative_prompt", label: "Negative prompt" },
    { ...field, key: "context", label: "Context", scope: "load" },
    { ...field, key: "width", label: "Width" }, { ...field, key: "hidden", label: "Hidden", available: false },
  ] }]);
  vi.mocked(api.workflowRevisionSchema).mockResolvedValue({ ...schema, input_schema_json: { properties: {
    width: { type: "integer", default: 512, readOnly: true }, quality: { type: "number", default: 1, readOnly: true },
  } } });
  show();
  await reference();
  expect(screen.getAllByRole("checkbox").map((input) => input.parentElement?.textContent)).toEqual(["Enabled", "Include Steps"]);
});

it.each([true, false])("offers enlargement controls only for an adjustable schema (fixed %s)", async (fixed) => {
  vi.mocked(api.engines).mockResolvedValue([{ ...engine, settings: [field,
    { ...field, key: "upscale_factor", label: "Enlarge by", type: "number", default: 2, minimum: 1, maximum: 8 },
  ] }]);
  vi.mocked(api.workflowRevisionSchema).mockResolvedValue({ ...schema, input_schema_json: { properties: {
    upscale_factor: fixed
      ? { type: "number", readOnly: true, title: "Enlargement", "x-lm-atelier-kind": "upscale" }
      : { type: "number", default: 2, minimum: 1, maximum: 8, title: "Enlarge by", "x-lm-atelier-kind": "upscale" },
  } } });
  const { save } = show();
  await reference();
  if (fixed) {
    expect(screen.queryByRole("checkbox", { name: "Include Enlarge by" })).toBeNull();
    expect(screen.queryByRole("spinbutton", { name: "Enlarge by" })).toBeNull();
  } else {
    fireEvent.click(screen.getByRole("checkbox", { name: "Include Enlarge by" }));
    expect(screen.getByRole("spinbutton", { name: "Enlarge by" })).toHaveValue(2);
  }
  fireEvent.change(screen.getByRole("textbox", { name: "Recipe name" }), { target: { value: "Picture size" } });
  fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));
  expect(save.mock.calls[0][0].settings_json).toEqual(fixed ? {} : { upscale_factor: 2 });
});

it("refuses to reuse cached schema controls after a failed refresh without losing the draft", async () => {
  const { client, save } = show(existing);
  await reference();
  fireEvent.change(screen.getByRole("spinbutton", { name: "Steps" }), { target: { value: "30" } });
  vi.mocked(api.workflowRevisionSchema).mockRejectedValue(new Error("Schema read failed"));
  void client.invalidateQueries({ queryKey: ["workflows", "revision-schema"] });
  await screen.findByText("Schema read failed");
  expect(screen.queryByRole("checkbox", { name: "Include Steps" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));
  expect(save).not.toHaveBeenCalled();
  vi.mocked(api.workflowRevisionSchema).mockResolvedValue(schema);
  fireEvent.click(screen.getByRole("button", { name: "Retry setting controls" }));
  await waitFor(() => expect(screen.getByRole("spinbutton", { name: "Steps" })).toHaveValue(30));
});

it("does not offer mismatched revision responses as current setting definitions", async () => {
  vi.mocked(api.workflowRevisionSchema).mockResolvedValue({ ...schema, revision_id: "other-revision" });
  show();
  await screen.findByRole("option", { name: "Reference workflow" });
  fireEvent.change(screen.getByRole("combobox", { name: "Reference workflow for settings" }), { target: { value: "reference" } });
  await screen.findByText("Setting controls are unavailable for this reference.");
  expect(screen.queryByRole("checkbox", { name: "Include Steps" })).toBeNull();
});

function pagedReferences(failMore = false) {
  const rows = Array.from({ length: 51 }, (_, index) => ({ id: "reference-" + index,
    name: "Reference " + String(index).padStart(2, "0"), operation: "text_to_image", description: "",
    current_revision_id: "revision-" + index, revision_count: 1, created_at: "", updated_at: "" }));
  vi.mocked(api.workflowSummaries).mockImplementation(async (options?: {
    workflowIds?: string[]; offset?: number; limit?: number; search?: string;
  }) => {
    if (options?.workflowIds) return rows.filter(row => options.workflowIds?.includes(row.id));
    if (failMore && options?.offset) throw new Error("More references unavailable");
    const matching = rows.filter(row => row.name.toLowerCase().includes(options?.search?.toLowerCase() ?? ""));
    return matching.slice(options?.offset ?? 0, (options?.offset ?? 0) + (options?.limit ?? rows.length));
  });
  vi.mocked(api.workflowRevisionSchema).mockImplementation(async revisionId => ({ ...schema,
    revision_id: revisionId, workflow_id: revisionId.replace("revision-", "reference-") }));
}

it("pages recipe references and preserves the selection while searching", async () => {
  pagedReferences();
  show();
  await screen.findByRole("option", { name: "Reference 00" });
  expect(api.workflowSummaries).toHaveBeenCalledWith(expect.objectContaining({ limit: 50, operation: "text_to_image" }), expect.any(AbortSignal));
  expect(screen.queryByRole("option", { name: "Reference 50" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Load more workflows" }));
  await screen.findByRole("option", { name: "Reference 50" });
  const select = screen.getByRole("combobox", { name: "Reference workflow for settings" });
  fireEvent.change(select, { target: { value: "reference-50" } });
  await screen.findByRole("checkbox", { name: "Include Steps" });
  fireEvent.change(screen.getByRole("searchbox", { name: "Search reference workflows" }), { target: { value: "Reference 00" } });
  await waitFor(() => expect(api.workflowSummaries).toHaveBeenCalledWith(expect.objectContaining({ search: "Reference 00", offset: 0 }), expect.any(AbortSignal)));
  expect(select).toHaveValue("reference-50");
  expect(screen.getByRole("option", { name: "Reference 50" })).toBeInTheDocument();
  expect(screen.getByRole("checkbox", { name: "Include Steps" })).toBeInTheDocument();
});

it("keeps recipe settings when the next reference page fails", async () => {
  pagedReferences(true);
  show();
  await screen.findByRole("option", { name: "Reference 00" });
  fireEvent.change(screen.getByRole("combobox", { name: "Reference workflow for settings" }), { target: { value: "reference-0" } });
  await screen.findByRole("checkbox", { name: "Include Steps" });
  fireEvent.click(screen.getByRole("button", { name: "Load more workflows" }));
  await screen.findByText("More references unavailable");
  expect(screen.getByRole("checkbox", { name: "Include Steps" })).toBeInTheDocument();
  expect(screen.getByRole("combobox", { name: "Reference workflow for settings" })).toHaveValue("reference-0");
  pagedReferences();
  fireEvent.click(screen.getByRole("button", { name: "Retry workflows" }));
  await screen.findByRole("option", { name: "Reference 50" });
});

it("does not use a cached reference after its exact lookup reports it missing", async () => {
  vi.mocked(api.workflowSummaries).mockImplementation(async (options?: { workflowIds?: string[] }) => options?.workflowIds ? [] : [{
    id: "reference", name: "Reference workflow", operation: "text_to_image", description: "",
    current_revision_id: "revision", revision_count: 1, created_at: "", updated_at: "",
  }]);
  show();
  await screen.findByRole("option", { name: "Reference workflow" });
  fireEvent.change(screen.getByRole("combobox", { name: "Reference workflow for settings" }), { target: { value: "reference" } });
  await screen.findByText("Selected reference is unavailable. Choose another workflow.");
  expect(screen.queryByRole("checkbox", { name: "Include Steps" })).toBeNull();
  expect(screen.queryByRole("option", { name: "Reference workflow" })).toBeNull();
  expect(screen.getAllByRole("option").filter(option => (option as HTMLOptionElement).value === "reference")).toHaveLength(1);
});

const referenceSummary: WorkflowSummary = {
  id: "reference", name: "Reference workflow", operation: "text_to_image", description: "",
  current_revision_id: "revision", revision_count: 1, created_at: "", updated_at: "",
};

it("keeps one neutral selected option while the exact reference is loading", async () => {
  let resolveReference: (rows: WorkflowSummary[]) => void = () => {};
  vi.mocked(api.workflowSummaries).mockImplementation(async options => options?.workflowIds
    ? new Promise<WorkflowSummary[]>(resolve => { resolveReference = resolve; }) : [referenceSummary]);
  show();
  await screen.findByRole("option", { name: "Reference workflow" });
  const select = screen.getByRole("combobox", { name: "Reference workflow for settings" });
  fireEvent.change(select, { target: { value: "reference" } });
  await screen.findByText("Loading selected reference…");
  expect(select).toHaveValue("reference");
  expect(screen.queryByRole("option", { name: /unavailable/ })).toBeNull();
  expect(screen.getByRole("option", { name: "Selected reference workflow" })).toBeInTheDocument();
  expect(screen.getAllByRole("option").filter(option => (option as HTMLOptionElement).value === "reference")).toHaveLength(1);
  expect(api.workflowRevisionSchema).not.toHaveBeenCalled();
  await waitFor(() => expect(api.workflowSummaries).toHaveBeenCalledWith(
    { workflowIds: ["reference"], limit: 1 }, expect.any(AbortSignal)));
  resolveReference([referenceSummary]);
  await screen.findByRole("checkbox", { name: "Include Steps" });
  expect(screen.getByRole("option", { name: "Reference workflow" })).toBeInTheDocument();
});

it("keeps the selected reference neutral after its exact lookup fails and recovers on retry", async () => {
  let failing = true;
  vi.mocked(api.workflowSummaries).mockImplementation(async options => {
    if (options?.workflowIds && failing) throw new Error("Exact reference read failed");
    return [referenceSummary];
  });
  show();
  await screen.findByRole("option", { name: "Reference workflow" });
  const select = screen.getByRole("combobox", { name: "Reference workflow for settings" });
  fireEvent.change(select, { target: { value: "reference" } });
  await screen.findByText("Exact reference read failed");
  expect(select).toHaveValue("reference");
  expect(screen.queryByRole("option", { name: /unavailable/ })).toBeNull();
  expect(screen.queryByRole("option", { name: "Reference workflow" })).toBeNull();
  expect(screen.getByRole("option", { name: "Selected reference workflow" })).toBeInTheDocument();
  expect(api.workflowRevisionSchema).not.toHaveBeenCalled();
  failing = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry setting controls" }));
  await screen.findByRole("checkbox", { name: "Include Steps" });
  expect(select).toHaveValue("reference");
});

it("keeps verified reference controls and edited settings when reference search fails", async () => {
  vi.mocked(api.workflowSummaries).mockImplementation(async options => {
    if (options?.search) throw new Error("Reference search failed");
    return [referenceSummary];
  });
  const { save } = show(existing);
  await reference();
  fireEvent.change(screen.getByRole("spinbutton", { name: "Steps" }), { target: { value: "30" } });
  fireEvent.change(screen.getByRole("searchbox", { name: "Search reference workflows" }), { target: { value: "Other" } });
  await screen.findByRole("button", { name: "Retry workflows" });
  expect(screen.getAllByText("Reference search failed").length).toBeGreaterThan(0);
  expect(screen.getByRole("combobox", { name: "Reference workflow for settings" })).toHaveValue("reference");
  expect(screen.getByRole("option", { name: "Reference workflow" })).toBeInTheDocument();
  expect(screen.getByRole("spinbutton", { name: "Steps" })).toHaveValue(30);
  fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));
  expect(save.mock.calls[0][0].settings_json).toEqual({ steps: 30, older_control: 0.7 });
});
