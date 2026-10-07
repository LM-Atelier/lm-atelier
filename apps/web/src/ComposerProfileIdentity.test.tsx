import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { TurnEditor } from "./TurnEditor";
import { profilePages } from "./test/modelLibraryPageFixtures";
import type { ComposerDraft } from "./composerPromptSource";
import type { ChatDetail, EngineCapabilities, EngineRole, ModelProfile } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, profilesPage: vi.fn(), presetsPage: vi.fn(),
    workflowFamilies: vi.fn(), chatWorkflowSelections: vi.fn(), projectWorkflowSelections: vi.fn(),
    modelAssets: vi.fn(), workflowRevisionSourceFit: vi.fn(),
  } };
});
const chat: ChatDetail = { id: "identity-chat", title: "Paper boat", project_id: null, archived: false,
  pinned: false, routing_mode: "auto", confirm_uncertain_media: false, active_chat_profile_id: null,
  active_image_profile_id: "chosen", active_video_profile_id: null, active_head_message_id: null,
  created_at: "2026-09-30T12:00:00Z", updated_at: "2026-09-30T12:00:00Z", messages: [] };
const engine: EngineCapabilities = { engine: "mock", version: "1", roles: ["chat", "image", "video"], operations: [],
  formats: [], devices: [], streaming: false, tool_calling: false, healthy: true, details: {},
  settings: [{ key: "width", label: "Width", type: "integer", default: 512, minimum: 64, maximum: 4096,
    step: 64, choices: [], scope: "request", visibility: "basic", restart_required: false,
    available: true, unavailable_reason: null, help: "" }],
};
const profiles: ModelProfile[] = [
  { id: "chosen", name: "Off-page image profile", model_install_id: "install-chosen", role: "image", engine: "mock",
    use_case: "", load_settings_json: {}, request_settings_json: { width: 1024 }, is_default: false },
  { id: "default", name: "Image default", model_install_id: "install-default", role: "image", engine: "mock",
    use_case: "", load_settings_json: {}, request_settings_json: { width: 768 }, is_default: true },
];
const clients: QueryClient[] = [];
const ignore = () => {};
function Harness({ selected, frozen }: { selected: string | null; frozen?: Record<string, unknown> }) {
  const [role, setRole] = useState<EngineRole>("chat");
  const [draft, setDraft] = useState<ComposerDraft>({ text: "A paper boat", promptSource: null });
  return <TurnEditor chat={{ ...chat, active_image_profile_id: selected }} engines={[engine]}
    stoppable={false} settings={{}} onSettings={ignore} settingsRole={role} onSettingsRole={setRole}
    presetId={null} onPreset={ignore} onMode={ignore} onSend={ignore} onStop={ignore} onStopAndSend={ignore}
    maxMediaOutputsPerPlan={4} draft={draft} onDraftChange={setDraft} profileValuesOverride={frozen} />;
}
function mount(selected: string | null = "chosen", frozen?: Record<string, unknown>) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><Harness selected={selected} frozen={frozen} /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
}
function imageSettings() { fireEvent.click(screen.getByRole("button", { name: "image" })); }
beforeEach(() => {
  localStorage.clear();
  vi.mocked(api.profilesPage).mockImplementation((options) => profilePages({ profiles: async () => profiles }, options));
  vi.mocked(api.presetsPage).mockResolvedValue([]);
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.projectWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.modelAssets).mockResolvedValue([]);
  vi.mocked(api.workflowRevisionSourceFit).mockResolvedValue({ available: false, reason: "source_fit_workflow_unsupported", modes: [], request_authorized: false });
});
afterEach(() => { cleanup(); clients.splice(0).forEach(client => client.clear()); vi.resetAllMocks(); });

it.each([["chosen", 1024], [null, 768], ["__auto__", 768]] as const)(
  "resolves image profile %s even while another settings role is displayed", async (selected, expected) => {
    mount(selected);
    await waitFor(() => expect(api.profilesPage).toHaveBeenCalledWith(expect.objectContaining({ role: "image", limit: 1,
      ...(selected === "chosen" ? { profileIds: ["chosen"] } : { defaultsOnly: true }) })));
    expect(vi.mocked(api.profilesPage).mock.calls.every(([options]) => options.role === "image")).toBe(true);
    imageSettings();
    await waitFor(() => expect(screen.getByRole("spinbutton", { name: "Width" })).toHaveValue(expected));
  },
);

it("withholds image fields until the exact selected profile arrives", async () => {
  let finish!: (rows: ModelProfile[]) => void;
  vi.mocked(api.profilesPage).mockReturnValue(new Promise(resolve => { finish = resolve; }));
  mount(); imageSettings();
  await screen.findByText("Loading selected model settings…");
  expect(screen.queryByRole("spinbutton", { name: "Width" })).not.toBeInTheDocument();
  await act(async () => finish([profiles[0]]));
  await waitFor(() => expect(screen.getByRole("spinbutton", { name: "Width" })).toHaveValue(1024));
});

it.each(["missing", "failed"])("does not substitute the default when the selected profile is %s", async (state) => {
  vi.mocked(api.profilesPage).mockImplementation(async (options) => {
    if (options.profileIds) {
      if (state === "failed") throw new Error("Image profile lookup failed");
      return [];
    }
    return [profiles[1]];
  });
  mount(); imageSettings();
  await screen.findByRole("button", { name: "Retry model settings" });
  expect(screen.queryByRole("spinbutton", { name: "Width" })).not.toBeInTheDocument();
  expect(api.profilesPage).not.toHaveBeenCalledWith(expect.objectContaining({ defaultsOnly: true }));
  vi.mocked(api.profilesPage).mockResolvedValue([profiles[0]]);
  fireEvent.click(screen.getByRole("button", { name: "Retry model settings" }));
  await waitFor(() => expect(screen.getByRole("spinbutton", { name: "Width" })).toHaveValue(1024));
});

it("keeps a supplied frozen profile independent of the current image selection", async () => {
  mount("chosen", { width: 896 }); imageSettings();
  await waitFor(() => expect(screen.getByRole("spinbutton", { name: "Width" })).toHaveValue(896));
  expect(api.profilesPage).not.toHaveBeenCalled();
});
