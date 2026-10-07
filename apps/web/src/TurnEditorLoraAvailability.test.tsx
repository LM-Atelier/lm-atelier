import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import type { ComposerDraft } from "./composerPromptSource";
import { TurnEditor, type TurnEditorProps } from "./TurnEditor";
import type { ChatDetail, EngineCapabilities, WorkflowLoraControls } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, profilesPage: vi.fn().mockResolvedValue([]), workflowFamilies: vi.fn(), workflowLoraControls: vi.fn() } };
});

const revision = "wfrev_landscape";
const key = ["workflows", "lora-controls", revision];
const stack = [
  { asset_id: "watercolor", model_strength: 0.7, clip_strength: 0.7, enabled: true },
  { asset_id: "ink", model_strength: 0.4, clip_strength: 0.4, enabled: true },
];
const chat: ChatDetail = {
  id: "chat-landscape", project_id: null, title: "Landscape", archived: false, pinned: false,
  routing_mode: "image", confirm_uncertain_media: false, active_chat_profile_id: null,
  active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: null,
  created_at: "2026-09-28T12:00:00Z", updated_at: "2026-09-28T12:00:00Z", messages: [],
};
const engine: EngineCapabilities = {
  engine: "mock", version: "1", roles: ["image"], operations: [], formats: [], devices: [],
  streaming: false, tool_calling: false, healthy: true, details: {},
  settings: [{ key: "width", label: "Width", type: "integer", default: 512, minimum: 64, maximum: 2048,
    step: 64, choices: [], scope: "request", visibility: "basic", restart_required: false,
    available: true, unavailable_reason: null, help: "" }],
};
const clients: QueryClient[] = [];
const ignore = () => {};

async function controls(accepts = true): Promise<WorkflowLoraControls> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(revision));
  return {
    version: 1, override_contract_version: 1, strength_bounds: { minimum: -4, maximum: 4 },
    override_target: null,
    revision_scope_sha256: Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(""),
    api_graph_sha256: "a".repeat(64), dependency_contract_sha256: "b".repeat(64),
    activation_binding_sha256: null, ordering_authority: "presentation_only", evidence_gaps: [],
    base_model_family: null, accepts_added_loras: accepts, slots: [],
  };
}

function Harness({ send, selected = true, stoppable = false }: {
  send: TurnEditorProps["onSend"]; selected?: boolean; stoppable?: boolean;
}) {
  const [draft, setDraft] = useState<ComposerDraft>({ text: "Paint a landscape", promptSource: null });
  const [settings, setSettings] = useState<Record<string, unknown>>({ width: 640, ...(selected ? { loras: stack } : {}) });
  return <TurnEditor chat={chat} engines={[engine]}
    stoppable={stoppable} settings={settings} onSettings={setSettings} settingsRole="image" onSettingsRole={ignore}
    presetId={null} onPreset={ignore} onMode={ignore} onSend={send} onStop={ignore} onStopAndSend={send}
    maxMediaOutputsPerPlan={4} draft={draft} onDraftChange={setDraft}
    workflowControl={<span>Landscape workflow</span>}
    workflowSelection={{ selector_capability: "image", mode: "revision", workflow_revision_id: revision,
      workflow_family_id: null, legacy_profile_id: null }} />;
}

function mount(send: TurnEditorProps["onSend"], initial?: WorkflowLoraControls, selected = true, stoppable = false) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(["workflows", "revision-schema", revision], {
    revision_id: revision, workflow_id: "landscape", operation: "text_to_image",
    input_schema_json: { type: "object", properties: { width: { type: "integer", default: 512 } } },
  });
  if (initial) client.setQueryData(key, initial);
  render(<QueryClientProvider client={client}><Harness send={send} selected={selected} stoppable={stoppable} /></QueryClientProvider>);
  return client;
}

beforeEach(() => { vi.mocked(api.profilesPage).mockResolvedValue([]); vi.mocked(api.workflowFamilies).mockResolvedValue([]); });
afterEach(() => {
  cleanup();
  for (const client of clients.splice(0)) client.clear();
  vi.resetAllMocks();
});

it.each([false, true])("keeps a restored stack while controls load before sending, stoppable=%s", async (stoppable) => {
  let finish!: (value: WorkflowLoraControls) => void;
  vi.mocked(api.workflowLoraControls).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
  const send = vi.fn();
  const client = mount(send, undefined, true, stoppable);
  await waitFor(() => expect(api.workflowLoraControls).toHaveBeenCalledTimes(1));
  const button = screen.getByRole("button", { name: stoppable ? "Stop current response and send" : "Send" });
  fireEvent.click(button);
  expect(send).not.toHaveBeenCalled();
  expect(screen.getByLabelText("Message")).toHaveValue("Paint a landscape");
  expect(screen.getByRole("alert")).toHaveTextContent("LoRA");
  await act(async () => finish(await controls()));
  await waitFor(() => expect(client.getQueryState(key)?.status).toBe("success"));
  fireEvent.click(button);
  expect(send).toHaveBeenCalledTimes(1);
  expect(send.mock.calls[0][3]).toEqual({ width: 640, loras: stack });
});

it("retains the chosen stack after a failed controls refresh and sends it after retry", async () => {
  const send = vi.fn();
  const client = mount(send, await controls());
  vi.mocked(api.workflowLoraControls).mockRejectedValue(new Error("Unavailable"));
  await act(async () => { await client.refetchQueries({ queryKey: key }); });
  await waitFor(() => expect(client.getQueryState(key)?.status).toBe("error"));
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
  expect(send).not.toHaveBeenCalled();
  expect(screen.getByLabelText("Message")).toHaveValue("Paint a landscape");
  vi.mocked(api.workflowLoraControls).mockResolvedValue(await controls());
  fireEvent.click(screen.getByRole("button", { name: "Retry LoRA controls" }));
  await waitFor(() => expect(client.getQueryState(key)?.status).toBe("success"));
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
  expect(send).toHaveBeenCalledTimes(1);
  expect(send.mock.calls[0][3]).toEqual({ width: 640, loras: stack });
});

it("keeps authoritative filtering when the workflow cannot take added LoRAs", async () => {
  const send = vi.fn();
  mount(send, await controls(false));
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
  expect(send).toHaveBeenCalledTimes(1);
  expect(send.mock.calls[0][3]).toEqual({ width: 640 });
});

it("does not hold a send without a chosen stack while controls load", async () => {
  vi.mocked(api.workflowLoraControls).mockImplementation(() => new Promise(() => {}));
  const send = vi.fn();
  mount(send, undefined, false);
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
  expect(send).toHaveBeenCalledTimes(1);
  expect(send.mock.calls[0][3]).toEqual({ width: 640 });
});
