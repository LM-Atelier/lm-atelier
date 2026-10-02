import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { api, connectEvents } from "./api";
import type { Chat, ChatDetail, Message, TurnAccepted, WebSearch } from "./types";
import { applyAcceptedChatPages } from "./acceptedChatPages";
import { asChatSummary } from "./chatSummaryFixtures";

vi.mock("./api", () => ({
  api: {
    searchConfiguration: vi.fn(), setupReadiness: vi.fn(), projects: vi.fn(), chats: vi.fn(), chatSummaries: vi.fn(),
    chat: vi.fn(), chatMetadata: vi.fn(), chatMessages: vi.fn(), chatContext: vi.fn(), chatSearches: vi.fn(), chatEditLineage: vi.fn(),
    workPlans: vi.fn(), engines: vi.fn(), profiles: vi.fn(), profilesPage: vi.fn(), presets: vi.fn(), presetsPage: vi.fn(), workflows: vi.fn(),
    workflowFamilies: vi.fn(), chatWorkflowSelections: vi.fn(), projectWorkflowSelections: vi.fn(),
    about: vi.fn(), jobs: vi.fn(), system: vi.fn(), workers: vi.fn(), runtimes: vi.fn(), backups: vi.fn(),
    editedBranches: vi.fn(), composerDraft: vi.fn(), saveComposerDraft: vi.fn(), classifyDraft: vi.fn(),
  },
  connectEvents: vi.fn().mockResolvedValue(() => undefined),
}));

const stamp = "2026-09-01T00:00:00Z";
const messages: Message[] = Array.from({ length: 80 }, (_, index) => ({
  id: `page-message-${index}`, chat_id: "paged-chat", parent_id: index ? `page-message-${index - 1}` : null,
  role: "assistant", status: "complete", created_at: stamp, updated_at: stamp,
  parts: [{ id: `part-${index}`, position: 0, type: "text", text: `Transcript row ${index}`,
    artifact_id: null, metadata_json: {} }],
}));
const chat: ChatDetail = {
  id: "paged-chat", title: "A long conversation", project_id: null, pinned: false, archived: false,
  routing_mode: "auto", confirm_uncertain_media: true, active_chat_profile_id: null,
  active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: messages[79].id,
  created_at: stamp, updated_at: stamp, messages,
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.profilesPage).mockImplementation(async (options) => (await import("./test/modelLibraryPageFixtures")).profilePages(api, options));
  vi.mocked(api.presetsPage).mockImplementation(async (options) => (await import("./test/modelLibraryPageFixtures")).presetPages(api, options));
  vi.mocked(connectEvents).mockResolvedValue(() => undefined);
  localStorage.clear(); sessionStorage.clear();
  localStorage.setItem("local-lm-chat", chat.id);
  Element.prototype.scrollIntoView = vi.fn();
  vi.mocked(api.setupReadiness).mockResolvedValue({ version: 2, state: "ready", roles: [] });
  vi.mocked(api.searchConfiguration).mockResolvedValue({ installation_enabled: false, configured: false,
    provider: "CRW", provider_endpoint: null, error_code: "search_not_configured" });
  vi.mocked(api.system).mockResolvedValue(null as never);
  vi.mocked(api.about).mockResolvedValue({ max_media_outputs_per_plan: 8 } as never);
  for (const list of [api.projects, api.workPlans, api.engines, api.profiles, api.presets, api.workflows,
    api.workflowFamilies, api.chatWorkflowSelections, api.projectWorkflowSelections,
    api.jobs, api.workers, api.runtimes, api.backups]) vi.mocked(list).mockResolvedValue([]);
  vi.mocked(api.chats).mockResolvedValue([chat]);
  vi.mocked(api.chatSummaries).mockResolvedValue([asChatSummary(chat)]);
  vi.mocked(api.chat).mockResolvedValue(chat);
  const metadata: Partial<ChatDetail> = { ...chat };
  delete metadata.messages;
  vi.mocked(api.chatMetadata).mockResolvedValue(metadata as Chat);
  vi.mocked(api.chatMessages).mockImplementation(async (_id, options) => ({
    chat_id: chat.id, messages: options?.before ? messages.slice(0, 40) : messages.slice(40),
    has_older: !options?.before, has_newer: Boolean(options?.before),
  }));
  vi.mocked(api.chatContext).mockResolvedValue({ chat_id: chat.id, head_id: messages[79].id,
    has_prior_image: false, has_prior_visual: false, has_pending_response: false });
  vi.mocked(api.chatSearches).mockResolvedValue({ chat_id: chat.id, searches: [], next_before: null });
  vi.mocked(api.editedBranches).mockResolvedValue({ items: [], next_cursor: null });
  vi.mocked(api.composerDraft).mockResolvedValue({ chat_id: chat.id, revision: 0, updated_at: null,
    text: "", prompt_source: null, mode: "auto", output_count: 1, attachments: [], mentions: [], template_settings: null });
});

afterEach(() => {
  cleanup();
  delete (Element.prototype as { scrollIntoView?: unknown }).scrollIntoView;
});

function open() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
  return client;
}

it("opens only the latest page and makes the oldest messages reachable", async () => {
  open();
  await screen.findByText("Transcript row 79");
  expect(screen.queryByText("Transcript row 0")).not.toBeInTheDocument();
  expect(api.chat).not.toHaveBeenCalled();
  expect(api.chatMessages).toHaveBeenCalledWith(chat.id, expect.objectContaining({ limit: 40, headId: messages[79].id }));

  const viewport = screen.getByText("Transcript row 79").closest(".messages") as HTMLElement;
  Object.defineProperty(viewport, "scrollHeight", { configurable: true,
    get: () => screen.queryByText("Transcript row 0") ? 2000 : 1000 });
  viewport.scrollTop = 120;
  const load = screen.getByRole("button", { name: "Load older messages" });
  load.focus();
  vi.mocked(api.chatMessages).mockRejectedValueOnce(new Error("Older messages unavailable"));
  fireEvent.click(load);
  await screen.findByText("Older messages unavailable");
  expect(viewport.scrollTop).toBe(120);
  expect(screen.getByText("Transcript row 79")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Load older messages" }));
  await screen.findByText("Transcript row 0");
  expect(screen.getAllByText("Transcript row 79")).toHaveLength(1);
  expect(screen.getByRole("button", { name: "All messages loaded" })).toHaveAttribute("aria-disabled", "true");
  expect(screen.getByRole("button", { name: "All messages loaded" })).toHaveFocus();
  expect(viewport.scrollTop).toBe(1120);
}, 15_000);

it("keeps image comparisons when the source turn is outside the loaded page", async () => {
  vi.mocked(api.chatMessages).mockResolvedValue({ chat_id: chat.id,
    messages: [{ ...messages[79], parts: [{ id: "image-result", position: 0, type: "image", text: null,
      artifact_id: "result-image", metadata_json: {} }] }], has_older: true, has_newer: false });
  vi.mocked(api.chatEditLineage).mockResolvedValue({ chat_id: chat.id, result_message_id: messages[79].id,
    steps: [{ message_id: "off-page-user", artifact_id: "source-image", instruction: "Add contrast" }], next_before: null });
  open();
  fireEvent.click(await screen.findByRole("button", { name: "Compare with the source" }));
  expect(screen.getByAltText("The source before the edit")).toHaveAttribute("src", "/api/artifacts/source-image/content");
  expect(api.chat).not.toHaveBeenCalled();
  expect(api.chatEditLineage).toHaveBeenCalledWith(chat.id, messages[79].id, expect.objectContaining({ limit: 2 }));
});

it("renders every branch-scoped row when a linking parent is on another page", async () => {
  const ids = [...Array.from({ length: 39 }, (_, index) => `m-${String(index).padStart(2, "0")}`), "a-gap", "z-head"];
  const branch = ids.map((id, index): Message => ({ ...messages[index], id,
    parent_id: index ? ids[index - 1] : null,
    parts: [{ ...messages[index].parts[0], text: `Branch row ${id}` }] }));
  vi.mocked(api.chatMetadata).mockResolvedValue({ ...chat, active_head_message_id: "z-head" });
  vi.mocked(api.chatContext).mockResolvedValue({ chat_id: chat.id, head_id: "z-head",
    has_prior_image: false, has_prior_visual: false, has_pending_response: false });
  vi.mocked(api.chatMessages).mockImplementation(async (_id, options) => ({
    chat_id: chat.id, messages: options?.before ? [branch[39]] : [...branch.slice(0, 39), branch[40]],
    has_older: !options?.before, has_newer: Boolean(options?.before),
  }));
  open();
  await screen.findByText("Branch row z-head");
  expect(screen.getAllByText(/^Branch row /)).toHaveLength(40);
  expect(screen.queryByText("Branch row a-gap")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Load older messages" }));
  await screen.findByText("Branch row a-gap");
  expect(screen.getAllByText(/^Branch row /)).toHaveLength(41);
  expect(api.chatMessages).toHaveBeenLastCalledWith(chat.id, expect.objectContaining({ before: "m-00", limit: 40 }));
});

it("shows a failed metadata read and retries without a full-transcript fallback", async () => {
  vi.mocked(api.chatMetadata).mockRejectedValueOnce(new Error("Conversation unavailable"));
  open();
  await screen.findByText("Conversation unavailable");
  fireEvent.click(screen.getByRole("button", { name: "Retry conversation" }));
  await screen.findByText("Transcript row 79");
  expect(api.chat).not.toHaveBeenCalled();
});

it("keeps off-page stop controls and exposes older pending search decisions", async () => {
  vi.mocked(api.chatContext).mockResolvedValue({ chat_id: chat.id, head_id: messages[79].id,
    has_prior_image: true, has_prior_visual: true, has_pending_response: true });
  const search = (id: string): WebSearch => ({
    run_id: id, assistant_message_id: `other-${id}`, job_id: `job-${id}`, revision: 1,
    state: "awaiting_approval", query: `Materials for ${id}`, provider: "CRW",
    provider_endpoint: "https://search.example.test", dispatch_after: null,
    results: [], result_count: 0, truncated: false, error_code: null,
  });
  vi.mocked(api.chatSearches).mockImplementation(async (_id, options) => ({
    chat_id: chat.id, searches: options?.pendingOnly ? [search(options.before ? "older" : "newer")] : [],
    next_before: options?.pendingOnly && !options.before ? "newer" : null,
  }));
  open();
  await screen.findByRole("button", { name: "Stop current response" });
  await screen.findByDisplayValue("Materials for newer");
  const pager = screen.getByRole("button", { name: "Load more pending searches" });
  pager.focus();
  fireEvent.click(pager);
  await screen.findByDisplayValue("Materials for older");
  expect(pager).toHaveFocus();
  expect(pager).toHaveTextContent("All pending searches loaded");
  expect(pager).toHaveAttribute("aria-disabled", "true");
  const calls = vi.mocked(api.chatSearches).mock.calls.length;
  fireEvent.click(pager);
  expect(api.chatSearches).toHaveBeenCalledTimes(calls);
  await waitFor(() => expect(api.chatSearches).toHaveBeenCalledWith(chat.id,
    expect.objectContaining({ pendingOnly: true, before: "newer", limit: 40 })));
});

it("uses the sidebar for the chat name and keeps only history controls above its transcript", async () => {
  const client = open();
  const conversation = await screen.findByRole("region", { name: chat.title });
  expect(screen.queryByRole("heading", { name: chat.title })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: chat.title })).toBeVisible();
  expect(conversation.querySelector(".chat-header")).toBeNull();
  expect(screen.queryByText("Unfiled chat")).not.toBeInTheDocument();
  expect(screen.queryByText("Web access")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Load older messages" })).toHaveClass("secondary");
  client.clear();
});

it("keeps history controls out of an empty conversation", async () => {
  vi.mocked(api.chatMessages).mockResolvedValue({ chat_id: chat.id, messages: [], has_older: false, has_newer: false });
  const client = open();
  await screen.findByRole("heading", { name: "What should we make?" });
  expect(screen.queryByRole("button", { name: "All messages loaded" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Load older messages" })).not.toBeInTheDocument();
  client.clear();
});

it("keeps loaded pages, reader position and the image comparison as multiple outputs arrive", async () => {
  const image = { ...messages[79], parts: [{ id: "viewed-image", position: 0, type: "image" as const,
    text: null, artifact_id: "viewed-result", metadata_json: {} }] };
  const user = { ...messages[0], id: "batch-user", parent_id: image.id, role: "user" as const,
    parts: [{ ...messages[0].parts[0], text: "Draw three blue circles" }] };
  const outputs = [0, 1, 2].map((index): Message => ({ ...image, id: `batch-output-${index}`,
    parent_id: index ? `batch-output-${index - 1}` : user.id, status: "pending", parts: [] }));
  let all = [...messages.slice(0, 79), image];
  vi.mocked(api.chatMessages).mockImplementation(async (_id, options) => {
    const end = options?.before ? all.findIndex((item) => item.id === options.before) : all.length;
    const start = Math.max(0, end - (options?.limit ?? 40));
    return { chat_id: chat.id, messages: all.slice(start, end), has_older: start > 0, has_newer: end < all.length };
  });
  vi.mocked(api.chatEditLineage).mockResolvedValue({ chat_id: chat.id, result_message_id: image.id,
    steps: [{ message_id: "image-source", artifact_id: "viewed-source", instruction: "Add contrast" }], next_before: null });
  const client = open();
  const compare = await screen.findByRole("button", { name: "Compare with the source" });
  const viewport = compare.closest(".messages") as HTMLElement;
  Object.defineProperties(viewport, {
    scrollHeight: { configurable: true, value: 2000 }, clientHeight: { configurable: true, value: 400 },
  });
  fireEvent.click(screen.getByRole("button", { name: "Load older messages" }));
  await screen.findByText("Transcript row 0");
  viewport.scrollTop = 500;
  fireEvent.scroll(viewport);
  fireEvent.click(compare);
  const dialog = screen.getByRole("dialog", { name: "Compare with the source" });
  fireEvent.change(screen.getByRole("slider", { name: "Comparison position" }), { target: { value: "63" } });
  const head = outputs.at(-1)!.id;
  all = [...all, user, ...outputs];
  vi.mocked(api.chatMetadata).mockResolvedValue({ ...chat, active_head_message_id: head });
  vi.mocked(api.chatContext).mockResolvedValue({ chat_id: chat.id, head_id: head,
    has_prior_image: true, has_prior_visual: true, has_pending_response: true });
  act(() => applyAcceptedChatPages(client, chat.id, {
    user_message: user, assistant_message: outputs[0], assistant_messages: outputs, run: {},
  } as TurnAccepted, true));
  await act(async () => { await client.invalidateQueries({ queryKey: ["chat", chat.id] }); });
  expect(screen.getByText("Transcript row 0").closest(".messages") === viewport).toBe(true);
  expect(viewport.scrollTop).toBe(500);
  expect(screen.getByRole("dialog", { name: "Compare with the source" })).toBe(dialog);
  expect(screen.getByRole("slider", { name: "Comparison position" })).toHaveValue("63");
  expect(viewport.querySelectorAll(":scope > article.message")).toHaveLength(84);

  all = all.map((item) => item.id.startsWith("batch-output-") ? { ...item, status: "complete", parts: [
    { id: `${item.id}-image`, position: 0, type: "image", text: null, artifact_id: `${item.id}-artifact`, metadata_json: {} },
  ] } : item);
  await act(async () => { await client.invalidateQueries({ queryKey: ["chat", chat.id] }); });
  await waitFor(() => expect(viewport.querySelectorAll("img[src*='batch-output-']:not(.media-backdrop)")).toHaveLength(3));
  expect(screen.getByText("Transcript row 0").closest(".messages") === viewport).toBe(true);
  expect(viewport.scrollTop).toBe(500);
  expect(screen.getByRole("dialog", { name: "Compare with the source" })).toBe(dialog);
  expect(api.chat).not.toHaveBeenCalled();
  expect(vi.mocked(api.chatMessages).mock.calls.every((call) => (call[1]?.limit ?? 40) <= 40)).toBe(true);
  client.clear();
}, 15_000);
