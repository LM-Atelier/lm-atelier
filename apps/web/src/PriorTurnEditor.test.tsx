import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";
import { readPriorTurnEditDraft } from "./priorTurnEditDraft";
import { PriorTurnEditor } from "./PriorTurnEditor";
import type { ChatDetail, EngineCapabilities, GenerationPreset, PriorTurnEditSource, SettingField } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api,
    workflowFamilies: vi.fn(), chatWorkflowSelections: vi.fn(), projectWorkflowSelections: vi.fn(),
    classifyDraft: vi.fn(), references: vi.fn(), modelAssets: vi.fn(),
    getPriorTurnEditSource: vi.fn(), queueEditedMessage: vi.fn(), updateChat: vi.fn(),
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
  localStorage.clear();
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
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
    <PriorTurnEditor chat={chat} messageId={source.source_user_message_id} engines={[engine]} profiles={[]}
      workflows={[]} presets={presets} maxMediaOutputsPerPlan={4} onAccepted={onAccepted} onClose={vi.fn()} />
  </QueryClientProvider>);
  await screen.findByRole("textbox", { name: "Message" });
  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
  return mounted;
}
async function submit() {
  fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
  fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
  await waitFor(() => expect(api.queueEditedMessage).toHaveBeenCalledTimes(1));
  await screen.findByText("Queue full");
  expect(api.updateChat).not.toHaveBeenCalled();
  return vi.mocked(api.queueEditedMessage).mock.calls[0][1];
}
it("keeps repeated image step settings separate through navigation and reopening", async () => {
  const first = { ...source, step_id: "first", ordinal: 0, depends_on: [], settings: { width: 640, steps: 4 },
    resolved_settings: { width: 640, steps: 4 } };
  const second = { ...first, step_id: "second", ordinal: 1, source_run_id: "second-run", settings: { width: 896, steps: 8 },
    resolved_settings: { width: 896, steps: 8 } };
  vi.mocked(api.getPriorTurnEditSource).mockResolvedValue({ ...source, original_mode: "auto", steps: [first, second] });
  const mounted = await mount();
  expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(4);
  fireEvent.change(screen.getByRole("spinbutton", { name: "steps" }), { target: { value: "6" } });
  fireEvent.blur(screen.getByRole("spinbutton", { name: "steps" }));
  fireEvent.change(screen.getByLabelText("Settings for this version"), { target: { value: "step:second" } });
  expect(screen.getByRole("spinbutton", { name: "width" })).toHaveValue(896);
  expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(8);
  fireEvent.change(screen.getByRole("spinbutton", { name: "steps" }), { target: { value: "10" } });
  fireEvent.blur(screen.getByRole("spinbutton", { name: "steps" }));
  fireEvent.change(screen.getByLabelText("Settings for this version"), { target: { value: "step:first" } });
  expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(6);
  mounted.unmount();
  await mount();
  expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(6);
  expect(await submit()).toMatchObject({ mode: "auto", settings: {}, step_overrides: {
    first: { settings: { steps: 6 } }, second: { settings: { steps: 10 } },
  } });
});
it("retains role settings when navigating away and back without changing Auto routing", async () => {
  vi.mocked(api.getPriorTurnEditSource).mockResolvedValue({ ...source, original_mode: "auto" });
  await mount();
  fireEvent.change(screen.getByRole("spinbutton", { name: "width" }), { target: { value: "768" } });
  fireEvent.change(screen.getByLabelText("Settings for this version"), { target: { value: "role:chat" } });
  fireEvent.change(screen.getByLabelText("Settings for this version"), { target: { value: "role:image" } });
  expect(screen.getByRole("spinbutton", { name: "width" })).toHaveValue(768);
  expect(await submit()).toMatchObject({ mode: "auto", settings: {}, role_overrides: { image: { settings: { width: 768 } } } });
});
it("shows a role preset on each step without restoring omitted original settings", async () => {
  const first = { ...source, step_id: "first", ordinal: 0, depends_on: [] };
  const second = { ...first, step_id: "second", ordinal: 1, source_run_id: "second-run" };
  vi.mocked(api.getPriorTurnEditSource).mockResolvedValue({ ...source, original_mode: "auto", steps: [first, second] });
  const original = presets[0].settings_json;
  presets[0].settings_json = { steps: 15 };
  try {
    await mount();
    fireEvent.change(screen.getByLabelText("Settings for this version"), { target: { value: "role:image" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Preset for this version" }), { target: { value: "preset-scene" } });
    fireEvent.change(screen.getByLabelText("Settings for this version"), { target: { value: "step:second" } });
    expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(15);
    expect(screen.getByRole("spinbutton", { name: "width" })).toHaveValue(512);
    const request = await submit();
    expect(request.role_overrides).toEqual({ image: { preset_id: "preset-scene", settings: { steps: 15 } } });
    expect(request.step_overrides).toBeUndefined();
  } finally { presets[0].settings_json = original; }
});
it("shows frozen source values and leaves untouched fields inherited, including a JSON blur", async () => {
  await mount();
  expect(screen.getByText("This version only")).toBeInTheDocument();
  expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(9);
  expect(screen.getByRole("combobox", { name: "Preset for this version" })).toHaveValue("inherit");
  fireEvent.blur(screen.getByRole("textbox", { name: "options" }));
  expect((await submit()).settings).toEqual({});
});
it("remembers a deliberate return to the source value without making other fields explicit", async () => {
  await mount();
  const input = screen.getByRole("spinbutton", { name: "width" });
  input.focus();
  fireEvent.change(input, { target: { value: "768" } });
  expect(screen.getByRole("spinbutton", { name: "width" })).toBe(input);
  expect(input).toHaveFocus();
  fireEvent.change(input, { target: { value: "640" } });
  expect((await submit()).settings).toEqual({ width: 640 });
});
it("lets the current LoRA selection be explicitly chosen even when its values match the source", async () => {
  await mount();
  fireEvent.click(screen.getByRole("button", { name: "Use current LoRA selection" }));
  expect((await submit()).settings).toEqual({ loras: [] });
});
it("can reselect the same preset identity with its current values", async () => {
  await mount();
  fireEvent.change(screen.getByRole("combobox", { name: "Preset for this version" }), { target: { value: "preset-scene" } });
  expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(99);
  expect(await submit()).toMatchObject({ preset_id: "preset-scene", settings: { width: 1024, steps: 99 } });
});
it("distinguishes explicit no-preset from inheriting the original preset", async () => {
  await mount();
  fireEvent.change(screen.getByRole("combobox", { name: "Preset for this version" }), { target: { value: "none" } });
  expect(await submit()).toMatchObject({ preset_id: null, settings: {} });
});
it("restores original settings and preset inheritance after local edits", async () => {
  await mount();
  fireEvent.change(screen.getByRole("spinbutton", { name: "width" }), { target: { value: "768" } });
  fireEvent.blur(screen.getByRole("spinbutton", { name: "width" }));
  fireEvent.change(screen.getByRole("combobox", { name: "Preset for this version" }), { target: { value: "preset-scene" } });
  fireEvent.click(screen.getByRole("button", { name: "Restore original settings" }));
  expect(screen.getByRole("spinbutton", { name: "width" })).toHaveValue(640);
  expect(screen.getByRole("spinbutton", { name: "steps" })).toHaveValue(9);
  const request = await submit();
  expect(request.settings).toEqual({});
  expect(request).not.toHaveProperty("preset_id");
});
it("retains same-value field intent when the saved draft is reopened", async () => {
  const mounted = await mount();
  const input = screen.getByRole("textbox", { name: "options" });
  fireEvent.change(input, { target: { value: '{ "quality": "high" }' } });
  fireEvent.blur(input);
  mounted.unmount();
  await mount();
  expect((await submit()).settings).toEqual({ options: { quality: "high" } });
});


function confirmationError(kind: "ordered" | "media") {
  return new ApiError(409, kind === "ordered" ? {
    code: "ordered_plan_confirmation_required", plan: { steps: [{ mode: "text" }, { mode: "video" }] },
    estimate: { video_duration_seconds: 4, estimated_bytes: 1024 },
  } : {
    code: "route_confirmation_required", plan: { operation: "text_to_video" },
    estimate: { duration_seconds: 4, estimated_intermediate_bytes: 1024 },
  }, "Confirmation required");
}
async function requestConfirmation(kind: "ordered" | "media") {
  vi.mocked(api.queueEditedMessage).mockRejectedValueOnce(confirmationError(kind));
  await mount();
  fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
  fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
  return screen.findByRole("button", { name: kind === "ordered" ? "Start plan" : "Start video" });
}
it.each(["ordered", "media"] as const)("keeps the complete edited %s request and key through confirmation", async (kind) => {
  const button = await requestConfirmation(kind);
  const first = structuredClone(vi.mocked(api.queueEditedMessage).mock.calls[0][1]);
  fireEvent.click(button);
  await screen.findByText("Queue full");
  expect(api.queueEditedMessage).toHaveBeenCalledTimes(2);
  expect(vi.mocked(api.queueEditedMessage).mock.calls[1][1]).toEqual({
    ...first, mode: kind === "ordered" ? "auto" : "video", confirm_media: true,
  });
  expect(readPriorTurnEditDraft(chat.id, source.source_user_message_id)?.pending?.request)
    .toEqual(vi.mocked(api.queueEditedMessage).mock.calls[1][1]);
  expect(onAccepted).not.toHaveBeenCalled();
  expect(api.updateChat).not.toHaveBeenCalled();
});

it("keeps the edited draft and makes no retry when confirmation is cancelled", async () => {
  await requestConfirmation("media");
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  await screen.findByText("Confirmation required");
  expect(api.queueEditedMessage).toHaveBeenCalledTimes(1);
  expect(onAccepted).not.toHaveBeenCalled();
  expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue(source.text);
  expect(readPriorTurnEditDraft(chat.id, source.source_user_message_id)?.pending?.request.confirm_media).toBeUndefined();
});

it("reuses the confirmed wire request after a lost response and reopening, accepting once", async () => {
  fireEvent.click(await requestConfirmation("media"));
  await screen.findByText("Queue full");
  const confirmed = structuredClone(vi.mocked(api.queueEditedMessage).mock.calls[1][1]);
  cleanup();
  const accepted: Awaited<ReturnType<typeof api.queueEditedMessage>> = {
    source_run_id: source.source_run_id, source_message_id: source.source_user_message_id,
    branch_activated: false, work_plan_id: "edit-plan", branch_head_message_id: "edit-assistant",
    accepted_context_sha256: "b".repeat(64),
    run: { id: "edit-run", idempotency_key: confirmed.idempotency_key, chat_id: chat.id,
      user_message_id: "edit-user", assistant_message_id: "edit-assistant", operation: "text_to_video",
      status: "queued", standalone_prompt: source.text, profile_id: null, workflow_revision_id: null,
      settings_json: {}, provenance_json: {}, error: null, created_at: stamp, updated_at: stamp,
      started_at: null, completed_at: null, duration_ms: null },
    user_message: { id: "edit-user", chat_id: chat.id, parent_id: null, role: "user", status: "complete",
      created_at: stamp, updated_at: stamp, parts: [] },
    assistant_message: { id: "edit-assistant", chat_id: chat.id, parent_id: "edit-user", role: "assistant",
      status: "pending", created_at: stamp, updated_at: stamp, parts: [] },
  };
  vi.mocked(api.queueEditedMessage).mockResolvedValueOnce(accepted);
  await mount();
  fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
  fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
  await waitFor(() => expect(onAccepted).toHaveBeenCalledTimes(1));
  expect(onAccepted).toHaveBeenCalledWith(accepted);
  expect(api.queueEditedMessage).toHaveBeenCalledTimes(3);
  expect(vi.mocked(api.queueEditedMessage).mock.calls[2][1]).toEqual(confirmed);
  expect(readPriorTurnEditDraft(chat.id, source.source_user_message_id)).toBeNull();
  expect(screen.queryByRole("button", { name: "Start video" })).not.toBeInTheDocument();
});

it("does not ask again or make a third submission when the confirmed retry is refused", async () => {
  const button = await requestConfirmation("media");
  vi.mocked(api.queueEditedMessage).mockRejectedValueOnce(confirmationError("media"));
  fireEvent.click(button);
  await screen.findByText("Confirmation required");
  expect(api.queueEditedMessage).toHaveBeenCalledTimes(2);
  expect(screen.queryByRole("button", { name: "Start video" })).not.toBeInTheDocument();
  expect(onAccepted).not.toHaveBeenCalled();
});
