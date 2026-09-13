/** A chat's unsent draft outlasts the page: stored in the workspace, and back after a restart. */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import type { Artifact, Chat, ChatComposerDraft, ChatDetail, TurnAccepted } from "./types";

vi.mock("./api", () => ({
  api: {
    searchConfiguration: vi.fn(),
    setupReadiness: vi.fn(),
    projects: vi.fn(),
    chats: vi.fn(),
    chat: vi.fn(),
    workPlans: vi.fn(),
    engines: vi.fn(),
    profiles: vi.fn(),
    presets: vi.fn(),
    workflows: vi.fn(),
    workflowFamilies: vi.fn(),
    chatWorkflowSelections: vi.fn(),
    projectWorkflowSelections: vi.fn(),
    classifyDraft: vi.fn(),
    about: vi.fn(),
    jobs: vi.fn(),
    system: vi.fn(),
    workers: vi.fn(),
    runtimes: vi.fn(),
    backups: vi.fn(),
    updateChat: vi.fn(),
    upload: vi.fn(),
    sendTurn: vi.fn(),
    composerDraft: vi.fn(),
    saveComposerDraft: vi.fn(),
    artifact: vi.fn(),
  },
  connectEvents: vi.fn().mockResolvedValue(() => undefined),
}));

const stamp = "2026-09-01T00:00:00Z";
const SOURCE_URL = "https://example.test/brass-and-aluminium";

const first: Chat = {
  id: "draft-chat-first",
  project_id: null,
  title: "Garden sketches",
  pinned: false,
  archived: false,
  routing_mode: "image",
  confirm_uncertain_media: false,
  active_chat_profile_id: null,
  active_image_profile_id: null,
  active_video_profile_id: null,
  active_head_message_id: "draft-answer",
  created_at: stamp,
  updated_at: stamp,
};
const second: Chat = { ...first, id: "draft-chat-second", title: "Kitchen notes", routing_mode: "auto", active_head_message_id: null };

const firstDetail: ChatDetail = {
  ...first,
  messages: [{
    id: "draft-answer",
    chat_id: first.id,
    parent_id: null,
    role: "assistant",
    status: "complete",
    created_at: stamp,
    updated_at: stamp,
    parts: [{ id: "draft-answer-text", position: 0, type: "text", text: "Here is what I found.", artifact_id: null, metadata_json: {} }],
  }],
  web_searches: [{
    run_id: "draft-search", assistant_message_id: "draft-answer", job_id: null, revision: null,
    state: "complete", query: "Compare brass and aluminium", provider: "CRW",
    provider_endpoint: "https://search.example.test", dispatch_after: null,
    results: [{ url: SOURCE_URL, title: "Material comparison", snippet: "A neutral summary" }],
    result_count: 1, truncated: false, error_code: null,
  }],
};

const sketch: Artifact = {
  id: "sha256:draft-sketch",
  sha256: "draft-sketch",
  kind: "input",
  media_type: "image/png",
  size_bytes: 5,
  original_name: "sketch.png",
  metadata_json: { origin: "uploaded", uploaded: true },
  created_at: stamp,
  url: "/api/artifacts/sha256%3Adraft-sketch/content",
} as Artifact;

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  localStorage.setItem("local-lm-chat", first.id);
  vi.mocked(api.searchConfiguration).mockResolvedValue({
    installation_enabled: false, configured: false, provider: "CRW", provider_endpoint: null, error_code: "search_not_configured",
  } as never);
  vi.mocked(api.setupReadiness).mockResolvedValue({ version: 2, state: "ready", roles: [] });
  vi.mocked(api.system).mockResolvedValue(null as never);
  vi.mocked(api.classifyDraft).mockResolvedValue({ references_prior_visual: false });
  vi.mocked(api.about).mockResolvedValue({
    max_media_outputs_per_plan: 8,
    version: "0.1.8",
    web_access_enabled: false,
    data_directory: "/lm-atelier/data",
    log_directory: "/lm-atelier/data/logs",
    artifact_directory: "/lm-atelier/data/artifacts",
    artifact_directory_requested: null,
  });
  vi.mocked(api.engines).mockResolvedValue([{
    engine: "mock", version: "1", roles: ["chat", "image", "video"], operations: ["text", "text_to_image", "text_to_video"],
    formats: ["mock"], devices: ["cpu:0"], streaming: true, tool_calling: true, settings: [], healthy: true, details: {},
  }] as never);
  for (const list of [
    api.projects, api.workPlans, api.profiles, api.presets, api.workflows, api.workflowFamilies,
    api.chatWorkflowSelections, api.projectWorkflowSelections, api.jobs, api.workers, api.runtimes, api.backups,
  ]) {
    vi.mocked(list).mockResolvedValue([]);
  }
  vi.mocked(api.chats).mockResolvedValue([first, second]);
  vi.mocked(api.chat).mockImplementation(async (id) => (id === first.id ? firstDetail : { ...second, messages: [] }));
  vi.mocked(api.updateChat).mockImplementation(async (id) => (id === first.id ? first : second) as never);
  vi.mocked(api.upload).mockResolvedValue(sketch);
  vi.mocked(api.sendTurn).mockResolvedValue({} as TurnAccepted);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
}

function sketchAttached() {
  return screen.queryByRole("button", { name: /Remove .*sketch\.png/ });
}

const stored: ChatComposerDraft = {
  chat_id: first.id,
  revision: 3,
  updated_at: stamp,
  text: "Plant the tulips before the frost",
  prompt_source: null,
  mode: "image",
  output_count: 2,
  attachments: [{ artifact_id: sketch.id, kind: "image", origin: "uploaded" }],
  mentions: [],
  template_settings: null,
};

it("brings back a draft saved before the workspace was last closed", async () => {
  vi.mocked(api.artifact).mockResolvedValue(sketch);
  vi.mocked(api.composerDraft).mockImplementation(async (chatId) =>
    chatId === first.id ? stored : { ...stored, chat_id: chatId, revision: 0, text: "", attachments: [], output_count: 1, mode: "auto" });
  renderApp();

  const composer = await screen.findByRole("textbox", { name: "Message" });
  await waitFor(() => expect(composer).toHaveValue("Plant the tulips before the frost"));
  expect(sketchAttached()).toBeVisible();
  expect(screen.getByRole("combobox", { name: "Number of outputs" })).toHaveValue("2");
});

it("saves what is typed to the workspace once typing pauses", async () => {
  vi.mocked(api.composerDraft).mockResolvedValue({ ...stored, revision: 0, text: "", attachments: [], output_count: 1, mode: "auto" });
  vi.mocked(api.saveComposerDraft).mockImplementation(async (chatId, expected, draft) => ({
    ...stored, ...draft, chat_id: chatId, revision: expected + 1,
  }));
  renderApp();

  fireEvent.change(await screen.findByRole("textbox", { name: "Message" }), { target: { value: "A bench by the pond" } });

  await waitFor(() => expect(api.saveComposerDraft).toHaveBeenCalled(), { timeout: 3000 });
  const [chatId, expected, draft] = vi.mocked(api.saveComposerDraft).mock.calls.at(-1)!;
  expect([chatId, expected, draft.text]).toEqual([first.id, 0, "A bench by the pond"]);
});
