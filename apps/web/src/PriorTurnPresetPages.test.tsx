import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { initializePriorTurnEditDraft, readPriorTurnEditDraft, writePriorTurnEditDraft } from "./priorTurnEditDraft";
import { presetPages } from "./test/modelLibraryPageFixtures";
import { PriorTurnEditor } from "./PriorTurnEditor";
import type { ChatDetail, EngineCapabilities, GenerationPreset, PriorTurnEditSource, SettingField } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, profilesPage: vi.fn().mockResolvedValue([]),
    presetsPage: vi.fn(), workflowRevisionChoices: vi.fn(), workflowRevisionSchema: vi.fn(), workflowFamilies: vi.fn(), chatWorkflowSelections: vi.fn(), projectWorkflowSelections: vi.fn(),
    classifyDraft: vi.fn(), references: vi.fn(), modelAssets: vi.fn(),
    getPriorTurnEditSource: vi.fn(), queueEditedMessage: vi.fn(), updateChat: vi.fn(),
    workflowRevisionSourceFit: vi.fn(), previewWorkflowRevisionSourceFit: vi.fn(), previewPriorTurnSourceFit: vi.fn(),
  } };
});
const stamp = "2026-09-07T12:00:00Z";
const chat: ChatDetail = { id: "settings-chat", project_id: null, title: "Paper boats",
  archived: false, pinned: false, routing_mode: "image", confirm_uncertain_media: false,
  active_chat_profile_id: null, active_image_profile_id: null, active_video_profile_id: null,
  active_head_message_id: null, created_at: stamp, updated_at: stamp, messages: [] };
function field(key: string, type: SettingField["type"], value: unknown): SettingField {
  return { key, label: key, type, default: value, minimum: null, maximum: null, step: null, choices: [], scope: "request", visibility: "basic",
    restart_required: false, available: true, unavailable_reason: null, help: "" };
}
const engine: EngineCapabilities = { engine: "mock", version: "1", roles: ["image"], operations: ["text_to_image"],
  formats: [], devices: [], streaming: false, tool_calling: false, healthy: true, details: {},
  settings: [field("width", "integer", 512), field("steps", "integer", 20), field("options", "object", {}),
    field("loras", "array", [])] };
const source: PriorTurnEditSource = { source_user_message_id: "source-message", source_run_id: "source-run",
  source_snapshot_sha256: "a".repeat(64), chat_id: chat.id, text: "A blue paper boat",
  mode: "image", operation: "text_to_image", input_artifact_ids: [], input_artifacts: [], references: [],
  settings: { width: 640, options: { quality: "high" }, loras: [] },
  resolved_settings: { width: 640, steps: 9, options: { quality: "high" }, loras: [] }, settings_role: "image",
  output_count: 1, profile_id: null, vision_profile_id: null, preset_id: "preset-scene",
  preset: { id: "preset-scene", name: "Scene", settings_json: { steps: 9 } }, model_selection: {},
  workflow_selection: { selector_capability: "image", mode: "legacy", workflow_family_id: null,
    workflow_revision_id: null, legacy_profile_id: null }, workflow_revision_id: null, workflow_schema: null,
  context_messages: [], context_visual_artifacts: [], profile_settings: {}, prompt_source: null };
const presets: GenerationPreset[] = [{ id: "preset-scene", name: "Scene today", role: "image",
  is_default: true, settings_json: { width: 1024, steps: 99 } }];
const onAccepted = vi.fn();
const clients: QueryClient[] = [];
beforeEach(() => {
  vi.mocked(api.profilesPage).mockResolvedValue([]);
  vi.mocked(api.presetsPage).mockImplementation((options) => presetPages({ presets: async () => presets }, options));
  vi.mocked(api.workflowRevisionChoices).mockResolvedValue([]);
  localStorage.clear();
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.workflowRevisionSourceFit).mockResolvedValue({
    available: false, reason: "source_fit_workflow_unsupported", modes: [], request_authorized: false,
  });
  vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.projectWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.modelAssets).mockResolvedValue([]);
  vi.mocked(api.getPriorTurnEditSource).mockResolvedValue(structuredClone(source));
  vi.mocked(api.queueEditedMessage).mockRejectedValue(new Error("Queue full"));
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); vi.resetAllMocks(); });
async function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  const mounted = render(<QueryClientProvider client={client}>
    <PriorTurnEditor chat={chat} messageId={source.source_user_message_id} engines={[engine]}
      maxMediaOutputsPerPlan={4} onAccepted={onAccepted} onClose={vi.fn()} />
  </QueryClientProvider>);
  await screen.findByRole("textbox", { name: "Message" });
  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
  return mounted;
}
async function submit() {
  await waitFor(() => expect(screen.queryByText("Loading selected preset…")).not.toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
  fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
  await waitFor(() => expect(api.queueEditedMessage).toHaveBeenCalledTimes(1));
  await screen.findByText("Queue full");
  expect(api.updateChat).not.toHaveBeenCalled();
  return vi.mocked(api.queueEditedMessage).mock.calls[0][1];
}
it("pages and searches presets while preserving the frozen original configuration", async () => {
  const choices: GenerationPreset[] = Array.from({ length: 51 }, (_, index) => ({
    ...presets[0], id: `choice-${index}`, name: `Choice ${index}`, is_default: false,
    settings_json: { steps: 60 + index },
  }));
  vi.mocked(api.presetsPage).mockImplementation((options) => presetPages({ presets: async () => choices }, options));
  await mount();
  await screen.findByRole("option", { name: "Choice 0" });
  expect(screen.queryByRole("option", { name: "Choice 50" })).not.toBeInTheDocument();
  expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(9);
  expect(api.presetsPage).not.toHaveBeenCalledWith(expect.objectContaining({ defaultsOnly: true }));
  fireEvent.click(screen.getByRole("button", { name: "More presets" }));
  await screen.findByRole("option", { name: "Choice 50" });
  fireEvent.change(screen.getByLabelText("Search presets for this version"), { target: { value: "Choice 50" } });
  await waitFor(() => expect(api.presetsPage).toHaveBeenCalledWith({ role: "image", limit: 50, offset: 0, search: "Choice 50" }));
  await waitFor(() => expect(screen.queryByRole("option", { name: "Choice 0" })).not.toBeInTheDocument());
  fireEvent.change(screen.getByLabelText("Preset for this version"), { target: { value: "choice-50" } });
  await waitFor(() => expect(api.presetsPage).toHaveBeenCalledWith({ role: "image", limit: 1, presetIds: ["choice-50"] }));
  expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(110);
  expect(await submit()).toMatchObject({ preset_id: "choice-50", settings: { steps: 110 } });
});

it("keeps frozen inheritance usable when current preset choices fail", async () => {
  vi.mocked(api.presetsPage).mockRejectedValue(new Error("Choices unavailable"));
  await mount();
  await screen.findByRole("button", { name: "Retry preset choices" });
  expect(screen.getByLabelText("Preset for this version")).toHaveValue("inherit");
  expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(9);
  const request = await submit();
  expect(request).not.toHaveProperty("preset_id");
  expect(request.settings).toEqual({});
});

it.each(["loading", "failed", "missing"])("retains a restored off-page preset when its exact lookup is %s", async (state) => {
  writePriorTurnEditDraft({ ...initializePriorTurnEditDraft(structuredClone(source)),
    presetChoice: { kind: "explicit", value: "preset-scene" }, settings: { steps: 44 }, explicitSettingsKeys: ["steps"],
  });
  let finish!: (rows: GenerationPreset[]) => void;
  vi.mocked(api.presetsPage).mockImplementation(async (options) => {
    if (!options.presetIds) return [];
    if (state === "loading") return new Promise((resolve) => { finish = resolve; });
    if (state === "failed") throw new Error("Selected preset could not be loaded");
    return [];
  });
  await mount();
  const select = screen.getByLabelText<HTMLSelectElement>("Preset for this version");
  expect(select).toHaveValue("preset-scene");
  if (state !== "loading") await screen.findByRole("button", { name: "Retry selected preset" });
  fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
  fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
  await screen.findByText("Wait for the selected preset to load, or choose another preset.");
  expect(api.queueEditedMessage).not.toHaveBeenCalled();
  expect(readPriorTurnEditDraft(chat.id, source.source_user_message_id)).toMatchObject({
    presetChoice: { kind: "explicit", value: "preset-scene" }, settings: { steps: 44 },
  });
  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
  if (state === "loading") await act(async () => finish(presets));
  else {
    vi.mocked(api.presetsPage).mockImplementation((options) => presetPages({ presets: async () => presets }, options));
    fireEvent.click(screen.getByRole("button", { name: "Retry selected preset" }));
  }
  await screen.findByRole("option", { name: "Scene today" });
  expect(screen.getByLabelText("Preset for this version")).toHaveValue("preset-scene");
  expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(44);
  expect(await submit()).toMatchObject({ preset_id: "preset-scene", settings: { steps: 44 } });
});

it("retries the next page without losing a deliberately selected preset", async () => {
  const choices: GenerationPreset[] = Array.from({ length: 50 }, (_, index) => ({ ...presets[0],
    id: `choice-${index}`, name: `Choice ${index}`, is_default: false,
  }));
  let failed = true;
  vi.mocked(api.presetsPage).mockImplementation((options) => {
    if (options.offset === 50 && failed) return Promise.reject(new Error("Next page unavailable"));
    return presetPages({ presets: async () => [...choices, ...presets] }, options);
  });
  await mount();
  await screen.findByRole("option", { name: "Choice 0" });
  fireEvent.change(screen.getByLabelText("Preset for this version"), { target: { value: "choice-0" } });
  fireEvent.click(screen.getByRole("button", { name: "More presets" }));
  await screen.findByRole("button", { name: "Retry preset choices" });
  expect(screen.getByLabelText("Preset for this version")).toHaveValue("choice-0");
  expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(99);
  failed = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry preset choices" }));
  await screen.findByRole("option", { name: "Scene today" });
  expect(screen.getByLabelText("Preset for this version")).toHaveValue("choice-0");
});
