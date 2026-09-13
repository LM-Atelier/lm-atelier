/** The send key, chosen in Settings and honoured by the prompt workshop as well as the composer. */

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import { SEND_KEY_KEY } from "./sendKey";
import type { Chat, TurnAccepted } from "./types";

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
    createPromptHelper: vi.fn(),
    promptHelper: vi.fn(),
    updatePromptHelper: vi.fn(),
    deletePromptHelper: vi.fn(),
    sendTurn: vi.fn(),
  },
  connectEvents: vi.fn().mockResolvedValue(() => undefined),
}));

const stamp = "2026-09-01T00:00:00Z";
const chat: Chat = {
  id: "send-key-chat",
  project_id: null,
  title: "Send key",
  pinned: false,
  archived: false,
  routing_mode: "text",
  confirm_uncertain_media: false,
  active_chat_profile_id: null,
  active_image_profile_id: null,
  active_video_profile_id: null,
  active_head_message_id: null,
  created_at: stamp,
  updated_at: stamp,
};
const helper = { ...chat, id: "send-key-helper", title: "Prompt workshop", archived: true, draft_prompt: "A blue cup", messages: [] };

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem("local-lm-chat", chat.id);
  Element.prototype.scrollIntoView = vi.fn();
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
  vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
  vi.mocked(api.createPromptHelper).mockResolvedValue(helper);
  vi.mocked(api.promptHelper).mockResolvedValue(helper);
  vi.mocked(api.updatePromptHelper).mockResolvedValue(helper);
  vi.mocked(api.deletePromptHelper).mockResolvedValue(undefined);
  vi.mocked(api.sendTurn).mockResolvedValue({} as TurnAccepted);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  delete (Element.prototype as { scrollIntoView?: unknown }).scrollIntoView;
});

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
}

/** Open the workshop on a draft, wait until it can send, and type an instruction. */
async function workshopInstruction() {
  renderApp();
  fireEvent.change(await screen.findByRole("textbox", { name: "Message" }), { target: { value: "A blue cup" } });
  fireEvent.click(screen.getByRole("button", { name: "Open prompt workshop" }));
  await screen.findByRole("dialog", { name: "Prompt workshop" });
  await waitFor(() => expect(api.createPromptHelper).toHaveBeenCalled());
  // Opening the workshop sends its own first instruction; only what follows is
  // the keystroke under test.
  await waitFor(() => expect(api.sendTurn).toHaveBeenCalled());
  vi.mocked(api.sendTurn).mockClear();
  const instruction = screen.getByRole("textbox", { name: "Prompt workshop instruction" });
  fireEvent.change(instruction, { target: { value: "Make it warmer" } });
  return instruction;
}

it("sends a workshop instruction on Enter by default", async () => {
  const instruction = await workshopInstruction();

  fireEvent.keyDown(instruction, { key: "Enter" });

  await waitFor(() => expect(api.sendTurn).toHaveBeenCalledWith("send-key-helper", "Make it warmer", "text", [], {}));
});

it("leaves Enter for a new line in the workshop and sends on Ctrl+Enter once that is chosen", async () => {
  localStorage.setItem(SEND_KEY_KEY, "mod-enter");
  const instruction = await workshopInstruction();

  expect(fireEvent.keyDown(instruction, { key: "Enter" })).toBe(true);
  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(api.sendTurn).not.toHaveBeenCalled();

  fireEvent.keyDown(instruction, { key: "Enter", ctrlKey: true });

  await waitFor(() => expect(api.sendTurn).toHaveBeenCalledTimes(1));
});

it("does not send a workshop instruction on the Enter that confirms composed text", async () => {
  const instruction = await workshopInstruction();

  fireEvent.keyDown(instruction, { key: "Enter", isComposing: true });
  fireEvent.keyDown(instruction, { key: "Enter", keyCode: 229 });
  await new Promise((resolve) => setTimeout(resolve, 50));

  expect(api.sendTurn).not.toHaveBeenCalled();
});

it("opens Settings on General, where the send key is chosen", async () => {
  renderApp();
  fireEvent.click(await screen.findByRole("button", { name: "Settings" }));
  await screen.findByRole("region", { name: "General" });
  const sendKey = screen.getByRole("group", { name: "Send key" });
  expect(within(sendKey).getByRole("button", { name: "Enter sends" }).getAttribute("aria-pressed")).toBe("true");

  fireEvent.click(within(sendKey).getByRole("button", { name: "Ctrl+Enter sends" }));

  expect(localStorage.getItem(SEND_KEY_KEY)).toBe("mod-enter");
  expect(within(sendKey).getByRole("button", { name: "Ctrl+Enter sends" }).getAttribute("aria-pressed")).toBe("true");
});
