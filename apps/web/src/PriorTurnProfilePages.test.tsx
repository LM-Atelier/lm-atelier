import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { initializePriorTurnEditDraft, readPriorTurnEditDraft, writePriorTurnEditDraft } from "./priorTurnEditDraft";
import { presetPages, profilePages } from "./test/modelLibraryPageFixtures";
import { PriorTurnEditor } from "./PriorTurnEditor";
import type { ChatDetail, EngineCapabilities, GenerationPreset, ModelProfile, PriorTurnEditSource, SettingField } from "./types";

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
  return mounted;
}
async function submit() {
  await waitFor(() => expect(screen.queryByText("Loading selected preset…")).not.toBeInTheDocument());
  const close = screen.queryByRole("button", { name: "Close settings" });
  if (close) fireEvent.click(close);
  fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
  await waitFor(() => expect(api.queueEditedMessage).toHaveBeenCalledTimes(1));
  await screen.findByText("Queue full");
  expect(api.updateChat).not.toHaveBeenCalled();
  return vi.mocked(api.queueEditedMessage).mock.calls[0][1];
}
function profile(index: number, vision = false): ModelProfile {
  return { id: `model-${index}`, name: `Model ${index}`, model_install_id: `install-${index}`, role: vision ? "chat" : "image",
    engine: "mock", is_default: false, use_case: "", input_modalities: vision ? ["text", "image"] : ["text"],
    load_settings_json: {}, request_settings_json: { width: 1024 } };
}

it("pages and searches models without losing an off-page selection", async () => {
  const choices = Array.from({ length: 51 }, (_, index) => profile(index));
  vi.mocked(api.profilesPage).mockImplementation((options) => profilePages({ profiles: async () => choices }, options));
  await mount();
  await screen.findByRole("option", { name: "Model 0" });
  expect(screen.queryByRole("option", { name: "Model 50" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "More models" }));
  await screen.findByRole("option", { name: "Model 50" });
  fireEvent.change(screen.getByLabelText("Model for this version"), { target: { value: "model-50" } });
  await waitFor(() => expect(api.profilesPage).toHaveBeenCalledWith(expect.objectContaining({ profileIds: ["model-50"], limit: 1, role: "image" })));
  await waitFor(() => expect(screen.queryByText("Loading selected model settings…")).not.toBeInTheDocument());
  fireEvent.change(screen.getByLabelText("Search models for this version"), { target: { value: "Model 1" } });
  await waitFor(() => expect(screen.queryByRole("option", { name: "Model 0" })).not.toBeInTheDocument());
  expect(screen.getByLabelText("Model for this version")).toHaveValue("model-50");
  expect(screen.getByRole("option", { name: "Model 50" })).toBeInTheDocument();
  expect(await submit()).toMatchObject({ profile_id: "model-50" });
});

it("keeps frozen model configuration usable when current choices fail", async () => {
  vi.mocked(api.profilesPage).mockRejectedValue(new Error("Model choices unavailable"));
  await mount();
  await screen.findByRole("button", { name: "Retry model choices" });
  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
  expect(screen.getByRole("spinbutton", { name: "width" })).toHaveValue(640);
  const request = await submit();
  expect(request).not.toHaveProperty("profile_id");
});

it.each(["loading", "failed", "missing"])("retains an explicit model while its exact read is %s", async (state) => {
  writePriorTurnEditDraft({ ...initializePriorTurnEditDraft(structuredClone(source)), profileChoice: { kind: "explicit", value: "model-99" } });
  let finish!: (rows: ModelProfile[]) => void;
  vi.mocked(api.profilesPage).mockImplementation(async (options) => {
    if (!options.profileIds) return [];
    if (state === "loading") return new Promise(resolve => { finish = resolve; });
    if (state === "failed") throw new Error("Selected model unavailable");
    return [];
  });
  await mount();
  expect(screen.getByLabelText("Model for this version")).toHaveValue("model-99");
  if (state !== "loading") await screen.findByRole("button", { name: "Retry model settings" });
  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
  expect(screen.queryByRole("spinbutton", { name: "width" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
  fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
  await screen.findByText("Wait for the selected model settings to load, or choose another model.");
  expect(api.queueEditedMessage).not.toHaveBeenCalled();
  expect(readPriorTurnEditDraft(chat.id, source.source_user_message_id)?.profileChoice).toEqual({ kind: "explicit", value: "model-99" });
  if (state === "loading") await act(async () => finish([profile(99)]));
  else {
    vi.mocked(api.profilesPage).mockResolvedValue([profile(99)]);
    fireEvent.click(screen.getByRole("button", { name: "Retry model settings" }));
  }
  await screen.findByRole("option", { name: "Model 99" });
  expect(await submit()).toMatchObject({ profile_id: "model-99" });
});

it("asks for vision-capable profile pages and resolves the selected identity separately", async () => {
  vi.mocked(api.getPriorTurnEditSource).mockResolvedValue({ ...source, mode: "text", settings_role: "chat", operation: "chat", preset_id: null, preset: null });
  const choices = [profile(1, true)];
  vi.mocked(api.profilesPage).mockImplementation((options) => profilePages({ profiles: async () => choices }, options));
  await mount();
  await waitFor(() => expect(api.profilesPage).toHaveBeenCalledWith({ role: "chat", inputModality: "image", limit: 50, offset: 0, search: "" }));
  const vision = screen.getByLabelText<HTMLSelectElement>("Vision model for this version");
  await waitFor(() => expect(Array.from(vision.options).some(option => option.value === "model-1")).toBe(true));
  fireEvent.change(vision, { target: { value: "model-1" } });
  await waitFor(() => expect(api.profilesPage).toHaveBeenCalledWith(expect.objectContaining({ role: "chat", inputModality: "image", profileIds: ["model-1"], limit: 1 })));
  await waitFor(() => expect(screen.queryByText("Loading selected model settings…")).not.toBeInTheDocument());
  expect(await submit()).toMatchObject({ vision_profile_id: "model-1" });
});

it("retains loaded models when a later page fails and supports retry", async () => {
  const choices = Array.from({ length: 51 }, (_, index) => profile(index));
  let failed = true;
  vi.mocked(api.profilesPage).mockImplementation((options) => {
    if (options.offset === 50 && failed) return Promise.reject(new Error("More models unavailable"));
    return profilePages({ profiles: async () => choices }, options);
  });
  await mount();
  fireEvent.click(await screen.findByRole("button", { name: "More models" }));
  await screen.findByRole("button", { name: "Retry model choices" });
  expect(screen.getByRole("option", { name: "Model 0" })).toBeInTheDocument();
  failed = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry model choices" }));
  await screen.findByRole("option", { name: "Model 50" });
});

it("resolves the current default separately when model browsing is unavailable", async () => {
  vi.mocked(api.profilesPage).mockImplementation(async (options) => {
    if (options.defaultsOnly) return [{ ...profile(99), is_default: true }];
    throw new Error("Model browsing unavailable");
  });
  await mount();
  await screen.findByRole("button", { name: "Retry model choices" });
  fireEvent.change(screen.getByLabelText("Model for this version"), { target: { value: "default" } });
  await waitFor(() => expect(api.profilesPage).toHaveBeenCalledWith(expect.objectContaining({ role: "image", limit: 1, defaultsOnly: true })));
  await waitFor(() => expect(screen.queryByText("Loading selected model settings…")).not.toBeInTheDocument());
  expect(screen.getByLabelText("Model for this version")).toHaveValue("default");
  expect(await submit()).toMatchObject({ profile_id: null });
});

it("does not require an unrelated displayed role's default to queue the frozen turn", async () => {
  vi.mocked(api.getPriorTurnEditSource).mockResolvedValue({ ...source, original_mode: "auto" });
  vi.mocked(api.profilesPage).mockImplementation(async (options) => {
    if (options.role === "chat" && options.defaultsOnly) throw new Error("Text model settings unavailable");
    return [];
  });
  await mount();
  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
  fireEvent.change(screen.getByLabelText("Settings for this version"), { target: { value: "role:chat" } });
  await screen.findAllByRole("button", { name: "Retry model settings" });
  const request = await submit();
  expect(request.mode).toBe("auto");
  expect(request).not.toHaveProperty("profile_id");
  expect(request).not.toHaveProperty("role_overrides");
});
