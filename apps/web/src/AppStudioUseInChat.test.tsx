import { installChatReadFixtures } from "./chatReadFixtures";
/** A picture in Image Studio goes to a chat's composer as a reference, and only once. */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import type { Artifact, Chat, ChatDetail } from "./types";
import { useStudioImage } from "./useStudioImage";

vi.mock("./api", () => ({
  api: {
    searchConfiguration: vi.fn(),
    setupReadiness: vi.fn(),
    projects: vi.fn(),
    chats: vi.fn(),
    chatSummaries: vi.fn(),
    chat: vi.fn(),
    workPlans: vi.fn(),
    engines: vi.fn(),
    profiles: vi.fn(), profilesPage: vi.fn(),
    presets: vi.fn(), presetsPage: vi.fn(),
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
    openStudioSession: vi.fn(),
    studioSession: vi.fn(),
    artifact: vi.fn(),
    editTemplates: vi.fn(),
    studioCapabilities: vi.fn(),
  },
  connectEvents: vi.fn().mockResolvedValue(() => undefined),
}));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

const stamp = "2026-09-01T00:00:00Z";

const harbour: Artifact = {
  id: "sha256:harbour",
  sha256: "harbour",
  kind: "image",
  media_type: "image/png",
  size_bytes: 5,
  original_name: "harbour.png",
  metadata_json: {},
  created_at: stamp,
  url: "/api/artifacts/sha256%3Aharbour/content",
} as Artifact;

const first: Chat = {
  id: "studio-chat-first",
  project_id: null,
  title: "Harbour pictures",
  pinned: false,
  archived: false,
  routing_mode: "auto",
  confirm_uncertain_media: false,
  active_chat_profile_id: null,
  active_image_profile_id: null,
  active_video_profile_id: null,
  active_head_message_id: "studio-answer",
  created_at: stamp,
  updated_at: stamp,
};
const second: Chat = { ...first, id: "studio-chat-second", title: "Kitchen notes", active_head_message_id: null };

const firstDetail: ChatDetail = {
  ...first,
  messages: [{
    id: "studio-answer",
    chat_id: first.id,
    parent_id: null,
    role: "assistant",
    status: "complete",
    created_at: stamp,
    updated_at: stamp,
    parts: [{
      id: "studio-answer-image", position: 0, type: "image", text: null, artifact_id: harbour.id,
      artifact: harbour, metadata_json: {},
    }],
  }],
} as ChatDetail;

beforeEach(() => {
  vi.mocked(api.profilesPage).mockImplementation(async (options) => (await import("./test/modelLibraryPageFixtures")).profilePages(api, options));
  vi.mocked(api.presetsPage).mockImplementation(async (options) => (await import("./test/modelLibraryPageFixtures")).presetPages(api, options));
  installChatReadFixtures();
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
    api.editTemplates,
  ]) {
    vi.mocked(list).mockResolvedValue([]);
  }
  vi.mocked(api.chats).mockResolvedValue([first, second]);
  vi.mocked(api.chatSummaries).mockResolvedValue([first, second].map((chat) => ({
    ...chat,
    activity: { active_work_count: 0, unresolved_failed_count: 0, last_output: null, last_failure: null },
  })));
  vi.mocked(api.chat).mockImplementation(async (id) => (id === first.id ? firstDetail : { ...second, messages: [] }));
  vi.mocked(api.updateChat).mockImplementation(async (id) => (id === first.id ? first : second) as never);
  const studio = { id: "chat-studio", messages: [] } as never;
  vi.mocked(api.openStudioSession).mockResolvedValue(studio);
  vi.mocked(api.studioSession).mockResolvedValue(studio);
  vi.mocked(api.artifact).mockResolvedValue(harbour);
  vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  const bitmap = { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap;
  vi.mocked(useStudioImage).mockReturnValue({ bitmap, error: null, reload: vi.fn() });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
}

function harbourAttached() {
  return screen.queryByRole("button", { name: /Remove .*harbour\.png/ });
}

async function openChat(chat: Chat) {
  fireEvent.click(screen.getByText(chat.title));
  expect(await screen.findByRole("region", { name: chat.title })).toBeInTheDocument();
}

it("attaches the studio's picture to the chat it came from, once", async () => {
  renderApp();
  fireEvent.click(await screen.findByRole("button", { name: "Open this image in the Image Studio" }));
  const useInChat = await screen.findByRole("button", { name: "Use in chat" });
  // It waits for the picture's details, so the chat can name it.
  await waitFor(() => expect(useInChat).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(useInChat);

  expect(await screen.findByRole("region", { name: first.title })).toBeInTheDocument();
  expect(await screen.findByRole("button", { name: /Remove .*harbour\.png/ })).toBeVisible();
  // A reference leaves the chat's mode as it was.
  expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("");

  fireEvent.click(harbourAttached()!);
  expect(harbourAttached()).toBeNull();
  await openChat(second);
  await openChat(first);
  expect(harbourAttached()).toBeNull();
});
