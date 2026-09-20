import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ChatSidebar } from "./ChatSidebar";
import { asChatSummary } from "./chatSummaryFixtures";
import { changeChatPages, useChatPages } from "./useChatPages";
import type { ChatDetail } from "./types";

vi.mock("./api", () => ({ api: { chatSummaries: vi.fn(), chat: vi.fn() } }));
const stamp = "2026-09-20T00:00:00Z";
const detail: ChatDetail = {
  id: "chat", title: "Color study", project_id: null, archived: false, pinned: false,
  routing_mode: "auto", confirm_uncertain_media: true, active_chat_profile_id: null,
  active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: null,
  vision_settings_json: { max_images: 7, max_video_frames: 9, verify_image_edits: true, compile_visual_prompts: false },
  created_at: stamp, updated_at: stamp, messages: [],
};
const summary = asChatSummary(detail);

function setup() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  return { client, wrapper };
}
function sidebar(onUpdateChat = vi.fn()) {
  return <ChatSidebar projects={[]} engines={[]} presets={[]} currentChatId="chat" view="chat"
    onChat={vi.fn()} onSetup={vi.fn()} onView={vi.fn()} onNewChat={vi.fn()} onNewProject={vi.fn()}
    onExportProject={vi.fn()} onImportProject={vi.fn()} onUpdateChat={onUpdateChat} onDeleteChat={vi.fn()}
    onUpdateProject={vi.fn()} onDeleteProject={vi.fn()}
    sidebar={{ width: 272, collapsed: false, setWidth: vi.fn(), toggle: vi.fn() }} />;
}
beforeEach(() => { vi.mocked(api.chatSummaries).mockResolvedValue([summary]); vi.mocked(api.chat).mockResolvedValue(detail); });
afterEach(cleanup);

it("loads exact chat settings before showing management controls", async () => {
  let resolve!: (chat: ChatDetail) => void;
  vi.mocked(api.chat).mockImplementation(() => new Promise((done) => { resolve = done; }));
  const onUpdate = vi.fn();
  render(sidebar(onUpdate), { wrapper: setup().wrapper });
  const manage = await screen.findByRole("button", { name: "Manage Color study" });
  manage.focus();
  fireEvent.click(manage);
  expect(await screen.findByRole("status")).toHaveTextContent("Loading chat settings");
  expect(screen.queryByLabelText(/Confirm uncertain media/)).not.toBeInTheDocument();
  expect(api.chat).toHaveBeenCalledWith("chat");
  await act(async () => resolve(detail));
  expect(await screen.findByLabelText(/Confirm uncertain media/)).toBeChecked();
  expect(screen.getByLabelText(/Review image edits/)).toBeChecked();
  expect(screen.getByLabelText(/Compose visual prompts/)).not.toBeChecked();
  fireEvent.change(screen.getByLabelText("Title"), { target: { value: "Renamed study" } });
  fireEvent.click(screen.getByRole("button", { name: "Save chat" }));
  expect(onUpdate).toHaveBeenCalledWith("chat", expect.objectContaining({
    title: "Renamed study", confirm_uncertain_media: true,
    vision_settings_json: detail.vision_settings_json,
  }));
  expect(manage).toHaveFocus();
});

it("offers retry after management detail fails without invented settings", async () => {
  vi.mocked(api.chat).mockRejectedValueOnce(new Error("Offline"));
  render(sidebar(), { wrapper: setup().wrapper });
  fireEvent.click(await screen.findByRole("button", { name: "Manage Color study" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Chat settings could not be loaded");
  expect(screen.queryByRole("button", { name: "Save chat" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByRole("button", { name: "Save chat" })).toBeVisible();
});

it("keeps full settings out of the summary cache after a chat mutation", async () => {
  const { client, wrapper } = setup();
  const { result } = renderHook(() => useChatPages(), { wrapper });
  await waitFor(() => expect(result.current.data).toHaveLength(1));
  act(() => changeChatPages(client, (row) => ({ ...row, ...detail, title: "Renamed" })));
  await waitFor(() => expect(result.current.data?.[0].title).toBe("Renamed"));
  expect(Object.keys(result.current.data![0]).sort()).toEqual(Object.keys(summary).sort());
  expect(result.current.data![0]).not.toHaveProperty("vision_settings_json");
  expect(result.current.data![0]).not.toHaveProperty("messages");
});

it("shows bounded active and failure counts without clearing unseen output on opening", async () => {
  vi.mocked(api.chatSummaries).mockResolvedValue([{ ...summary, activity: {
    active_work_count: 123, unresolved_failed_count: 2, last_failure: null,
    last_output: { id: "output", sequence: 1, message_id: "message", response_revision_id: "revision", occurred_at: stamp },
  } }]);
  render(sidebar(), { wrapper: setup().wrapper });
  expect(await screen.findByRole("img", { name: "123 active tasks" })).toHaveTextContent("99+");
  expect(screen.getByRole("img", { name: "2 unresolved failures" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: /Color study.*Unread output/ }));
  expect(screen.getByRole("img", { name: "Unread output" })).toBeVisible();
});
