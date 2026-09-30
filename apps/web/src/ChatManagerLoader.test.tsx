import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ChatManagerLoader } from "./ChatManagerLoader";
import type { Chat } from "./types";

vi.mock("./api", () => ({ api: { chat: vi.fn(), chatMetadata: vi.fn() } }));

const chat: Chat = {
  id: "chat-settings", title: "Harbor sketches", project_id: null, archived: true,
  pinned: false, routing_mode: "image", confirm_uncertain_media: false,
  active_chat_profile_id: null, active_image_profile_id: null, active_video_profile_id: null,
  active_head_message_id: "message-last", vision_settings_json: { verify_image_edits: true },
  created_at: "2026-09-01T00:00:00Z", updated_at: "2026-09-01T00:00:00Z",
};

beforeEach(() => {
  vi.mocked(api.chat).mockReset().mockResolvedValue({ ...chat, messages: [] });
  vi.mocked(api.chatMetadata).mockReset().mockResolvedValue(chat);
});
afterEach(cleanup);

function openManager(onSave = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <ChatManagerLoader chatId={chat.id} projects={[]} onClose={vi.fn()}
      onSave={onSave} onDelete={vi.fn()} />
  </QueryClientProvider>);
}

it("edits chat settings from metadata without fetching its transcript", async () => {
  const save = vi.fn();
  openManager(save);
  const title = await screen.findByLabelText("Title");
  expect(title).toHaveValue("Harbor sketches");
  expect(screen.getByRole("checkbox", { name: /Archived/ })).toBeChecked();
  expect(screen.getByRole("checkbox", { name: /Review image edits/ })).toBeChecked();
  fireEvent.change(title, { target: { value: "Coastal sketches" } });
  fireEvent.click(screen.getByRole("button", { name: "Save chat" }));
  expect(save).toHaveBeenCalledWith(expect.objectContaining({ title: "Coastal sketches", archived: true }));
  expect(api.chatMetadata).toHaveBeenCalledWith(chat.id, expect.any(AbortSignal));
  expect(api.chat).not.toHaveBeenCalled();
});

it("retries a failed metadata read without falling back to the full transcript", async () => {
  vi.mocked(api.chatMetadata).mockRejectedValueOnce(new Error("temporarily unavailable"));
  openManager();
  expect(await screen.findByRole("alert")).toHaveTextContent("Chat settings could not be loaded");
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByLabelText("Title")).toHaveValue("Harbor sketches");
  expect(api.chatMetadata).toHaveBeenCalledTimes(2);
  expect(api.chat).not.toHaveBeenCalled();
});
