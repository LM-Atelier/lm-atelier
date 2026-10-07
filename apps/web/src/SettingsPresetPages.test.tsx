import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { SettingsDrawer } from "./SettingsDrawer";
import type { EngineCapabilities, EngineRole, GenerationPreset } from "./types";

vi.mock("./api", () => ({ api: { presetsPage: vi.fn(), workflowLoraControls: vi.fn() } }));
const engine: EngineCapabilities = {
  engine: "mock", version: "1", roles: ["chat", "image", "video"], operations: [], formats: [], devices: [],
  streaming: false, tool_calling: false, healthy: true, details: {},
  settings: [{ key: "steps", label: "Steps", type: "integer", default: 3, minimum: 1, maximum: 100,
    step: 1, choices: [], scope: "request", visibility: "basic", restart_required: false,
    available: true, unavailable_reason: null, help: "" }],
};
const presets: GenerationPreset[] = [
  ...Array.from({ length: 51 }, (_, index): GenerationPreset => ({ id: `preset-${index}`, name: `Choice ${index}`, role: "image", settings_json: { steps: index + 1 }, is_default: false })),
  { id: "workspace", name: "Workspace default", role: "image", settings_json: { steps: 7 }, is_default: true },
  { id: "project", name: "Project setting", role: "image", settings_json: { steps: 11 }, is_default: false },
  { id: "selected", name: "Selected distant setting", role: "image", settings_json: { steps: 41 }, is_default: false },
  { id: "unicode", name: "Straße %_ distant", role: "image", settings_json: { steps: 23 }, is_default: false },
];
function Harness({ selected = null, inherited = null }: { selected?: string | null; inherited?: string | null }) {
  const [presetId, setPresetId] = useState(selected);
  const [role, setRole] = useState<EngineRole>("image");
  return <SettingsDrawer open onClose={vi.fn()} mode="auto" role={role} onRole={setRole}
    engines={[engine]} values={{}} onValues={vi.fn()} presetId={presetId} onPreset={setPresetId}
    inheritedPresetId={inherited} imageEdit={false} imageEditPrompt="" />;
}
function show(selected: string | null = null, inherited: string | null = null) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><Harness selected={selected} inherited={inherited} /></QueryClientProvider>);
  return client;
}
beforeEach(() => {
  vi.mocked(api.presetsPage).mockReset().mockImplementation(async (options) => {
    const rows = presets.filter((preset) => preset.role === options.role
      && (!options.defaultsOnly || preset.is_default)
      && (!options.presetIds || options.presetIds.includes(preset.id))
      && (!options.search || preset.name.toLowerCase().replaceAll("ß", "ss").includes(options.search.toLowerCase())));
    return rows.slice(options.offset ?? 0, (options.offset ?? 0) + options.limit);
  });
});
afterEach(cleanup);

it.each([
  [null, null, 7], [null, "project", 11], ["selected", "project", 41],
])("resolves selected %s and inherited %s independently of the first page", async (selected, inherited, expected) => {
  show(selected, inherited);
  await waitFor(() => expect(screen.getByRole("spinbutton", { name: "Steps" })).toHaveValue(expected));
  expect(api.presetsPage).toHaveBeenCalledWith({ role: "image", limit: 1, defaultsOnly: true });
  expect(api.presetsPage).toHaveBeenCalledWith({ role: "image", limit: 50, offset: 0, search: "" });
  if (selected) expect(screen.getByRole("combobox", { name: "image preset" })).toHaveValue(selected);
  if (inherited) expect(screen.getByRole("option", { name: "Inherit · Project setting" })).toBeInTheDocument();
});

it("pages choices without changing the selected settings and resets search offsets", async () => {
  show("selected");
  expect(await screen.findByRole("option", { name: "Choice 0" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "More presets" }));
  expect(await screen.findByRole("option", { name: "Choice 50" })).toBeInTheDocument();
  expect(screen.getByRole("spinbutton", { name: "Steps" })).toHaveValue(41);
  fireEvent.change(screen.getByRole("textbox", { name: "Search image presets" }), { target: { value: "STRASSE %_" } });
  expect(await screen.findByRole("option", { name: "Straße %_ distant" })).toBeInTheDocument();
  expect(api.presetsPage).toHaveBeenCalledWith({ role: "image", limit: 50, offset: 0, search: "STRASSE %_" });
  expect(screen.queryByRole("option", { name: "Choice 0" })).not.toBeInTheDocument();
  expect(screen.getByRole("combobox", { name: "image preset" })).toHaveValue("selected");
  expect(screen.getByRole("spinbutton", { name: "Steps" })).toHaveValue(41);
});

it("waits for exact preset settings instead of showing engine defaults", async () => {
  let finish!: (rows: GenerationPreset[]) => void;
  const read = vi.mocked(api.presetsPage).getMockImplementation()!;
  vi.mocked(api.presetsPage).mockImplementation((options) => options.presetIds
    ? new Promise((resolve) => { finish = resolve; }) : read(options));
  show("selected");
  expect(await screen.findByText("Loading preset settings…")).toBeInTheDocument();
  expect(screen.queryByRole("spinbutton", { name: "Steps" })).not.toBeInTheDocument();
  expect(screen.getByRole("combobox", { name: "image preset" })).toHaveValue("selected");
  await act(async () => { finish(presets.filter((preset) => preset.id === "selected")); });
  expect(await screen.findByRole("spinbutton", { name: "Steps" })).toHaveValue(41);
});

it("retains a missing selection without quietly selecting a default", async () => {
  show("missing");
  expect(await screen.findByText("A selected or inherited preset is unavailable.")).toBeInTheDocument();
  expect(screen.getByRole("combobox", { name: "image preset" })).toHaveValue("missing");
  expect(screen.queryByRole("spinbutton", { name: "Steps" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByRole("combobox", { name: "image preset" }), { target: { value: "" } });
  expect(await screen.findByRole("spinbutton", { name: "Steps" })).toHaveValue(7);
});

it("keeps the selected identity through an exact-read failure and retry", async () => {
  const read = vi.mocked(api.presetsPage).getMockImplementation()!;
  let failed = false;
  vi.mocked(api.presetsPage).mockImplementation(async (options) => {
    if (options.presetIds && !failed) { failed = true; throw new Error("Exact preset read failed"); }
    return read(options);
  });
  show("selected");
  expect(await screen.findByText("Exact preset read failed")).toBeInTheDocument();
  expect(screen.getByRole("combobox", { name: "image preset" })).toHaveValue("selected");
  expect(screen.queryByRole("spinbutton", { name: "Steps" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry preset settings" }));
  expect(await screen.findByRole("spinbutton", { name: "Steps" })).toHaveValue(41);
});

it("keeps loaded choices and selected settings when the next page fails", async () => {
  const read = vi.mocked(api.presetsPage).getMockImplementation()!;
  let failed = false;
  vi.mocked(api.presetsPage).mockImplementation(async (options) => {
    if (options.offset === 50 && !failed) { failed = true; throw new Error("More choices failed"); }
    return read(options);
  });
  show("selected");
  fireEvent.click(await screen.findByRole("button", { name: "More presets" }));
  expect(await screen.findByText("More choices failed")).toBeInTheDocument();
  expect(screen.getByRole("option", { name: "Choice 0" })).toBeInTheDocument();
  expect(screen.getByRole("spinbutton", { name: "Steps" })).toHaveValue(41);
  fireEvent.click(screen.getByRole("button", { name: "Retry preset choices" }));
  expect(await screen.findByRole("option", { name: "Choice 50" })).toBeInTheDocument();
});


it("keeps the preset control focused while exact settings arrive", async () => {
  let finish!: (rows: GenerationPreset[]) => void;
  const read = vi.mocked(api.presetsPage).getMockImplementation()!;
  vi.mocked(api.presetsPage).mockImplementation((options) => options.presetIds
    ? new Promise((resolve) => { finish = resolve; }) : read(options));
  show();
  await waitFor(() => expect(screen.getByRole("spinbutton", { name: "Steps" })).toHaveValue(7));
  const selector = screen.getByRole("combobox", { name: "image preset" });
  selector.focus();
  fireEvent.change(selector, { target: { value: "preset-0" } });
  expect(await screen.findByText("Loading preset settings…")).toBeInTheDocument();
  expect(screen.getByRole("combobox", { name: "image preset" })).toBe(selector);
  expect(selector).toHaveFocus();
  await act(async () => { finish(presets.filter((preset) => preset.id === "preset-0")); });
  expect(await screen.findByRole("spinbutton", { name: "Steps" })).toHaveValue(1);
  expect(screen.getByRole("combobox", { name: "image preset" })).toBe(selector);
  expect(selector).toHaveFocus();
});
