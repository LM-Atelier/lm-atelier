/** Following a conversation to its newest message without animating it for somebody who asked not to. */

import { cleanup, render, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import { MOTION_KEY } from "./theme";
import type { ChatDetail } from "./types";

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
    about: vi.fn(),
    jobs: vi.fn(),
    system: vi.fn(),
    workers: vi.fn(),
    runtimes: vi.fn(),
    backups: vi.fn(),
  },
  connectEvents: vi.fn().mockResolvedValue(() => undefined),
}));

const stamp = "2026-09-01T00:00:00Z";
const chat: ChatDetail = {
  id: "motion-chat",
  project_id: null,
  title: "Motion",
  pinned: false,
  archived: false,
  routing_mode: "auto",
  confirm_uncertain_media: true,
  active_chat_profile_id: null,
  active_image_profile_id: null,
  active_video_profile_id: null,
  active_head_message_id: "motion-answer",
  created_at: stamp,
  updated_at: stamp,
  messages: [{
    id: "motion-answer",
    chat_id: "motion-chat",
    parent_id: null,
    role: "assistant",
    status: "complete",
    created_at: stamp,
    updated_at: stamp,
    parts: [{ id: "motion-text", position: 0, type: "text", text: "A calm reply", artifact_id: null, metadata_json: {} }],
  }],
};

const scrollIntoView = vi.fn();

/** A computer whose reduced-motion setting the test decides; every other query answers no. */
function computerReducesMotion(reduced: boolean) {
  vi.stubGlobal("matchMedia", (media: string) => ({
    media,
    matches: media === "(prefers-reduced-motion: reduce)" && reduced,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  }));
}

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem("local-lm-chat", chat.id);
  Element.prototype.scrollIntoView = scrollIntoView;
  vi.mocked(api.setupReadiness).mockResolvedValue({ version: 2, state: "ready", roles: [] });
  vi.mocked(api.system).mockResolvedValue(null as never);
  vi.mocked(api.about).mockResolvedValue({
    max_media_outputs_per_plan: 8,
    version: "0.1.8",
    web_access_enabled: false,
    data_directory: "/lm-atelier/data",
    log_directory: "/lm-atelier/data/logs",
    artifact_directory: "/lm-atelier/data/artifacts",
    artifact_directory_requested: null,
  });
  for (const list of [
    api.projects, api.workPlans, api.engines, api.profiles, api.presets,
    api.workflows, api.workflowFamilies, api.chatWorkflowSelections,
    api.projectWorkflowSelections, api.jobs, api.workers, api.runtimes, api.backups,
  ]) {
    vi.mocked(list).mockResolvedValue([]);
  }
  vi.mocked(api.chats).mockResolvedValue([chat]);
  vi.mocked(api.chat).mockResolvedValue(chat);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
  delete (Element.prototype as { scrollIntoView?: unknown }).scrollIntoView;
});

async function scrolledWith(): Promise<ScrollBehavior | undefined> {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
  await waitFor(() => expect(scrollIntoView).toHaveBeenCalled());
  return (scrollIntoView.mock.calls.at(-1)?.[0] as ScrollIntoViewOptions | undefined)?.behavior;
}

it("glides to the newest message when nothing asks for less motion", async () => {
  computerReducesMotion(false);

  expect(await scrolledWith()).toBe("smooth");
});

it("jumps to the newest message when the computer asks for less motion", async () => {
  computerReducesMotion(true);

  expect(await scrolledWith()).toBe("auto");
});

it("jumps to the newest message when Reduce is chosen, whatever the computer says", async () => {
  computerReducesMotion(false);
  localStorage.setItem(MOTION_KEY, "reduce");

  expect(await scrolledWith()).toBe("auto");
});
