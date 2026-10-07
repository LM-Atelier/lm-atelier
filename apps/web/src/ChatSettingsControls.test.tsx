import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ChatManagerLoader } from "./ChatManagerLoader";
import type { Chat } from "./types";

vi.mock("./api", () => ({ api: { chatMetadata: vi.fn(), projects: vi.fn().mockResolvedValue([]),
  workflowUseCasePresets: vi.fn().mockResolvedValue([]), workflowUseCaseChoice: vi.fn().mockResolvedValue({ mode: "inherit" }),
  searchConfiguration: vi.fn().mockResolvedValue({ installation_enabled: true, configured: true,
    provider: "CRW", provider_endpoint: "https://search.example.test", error_code: null }), updateChat: vi.fn(),
} }));
afterEach(() => { cleanup(); vi.clearAllMocks(); });

it("offers recipes and Web permissions in chat settings and preserves unsaved title edits across permission saves", async () => {
  let chat: Chat = { id: "settings-chat", title: "Neutral conversation", project_id: null, archived: false, pinned: false,
    routing_mode: "auto", confirm_uncertain_media: false, active_chat_profile_id: null, active_image_profile_id: null,
    active_video_profile_id: null, active_head_message_id: null, created_at: "2026-10-01", updated_at: "2026-10-01",
    web_settings_json: { allow_url_fetch: false, allow_search: false, allow_search_without_asking: false } };
  vi.mocked(api.chatMetadata).mockResolvedValue(chat);
  vi.mocked(api.updateChat).mockImplementation(async (_id, updates) => { chat = { ...chat, ...updates }; return chat; });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const save = vi.fn();
  render(<QueryClientProvider client={client}><ChatManagerLoader chatId={chat.id} onClose={vi.fn()} onSave={save} onDelete={vi.fn()} /></QueryClientProvider>);
  const title = await screen.findByLabelText("Title");
  fireEvent.change(title, { target: { value: "Unsaved new title" } });
  const dialog = screen.getByRole("dialog", { name: "Chat settings" });
  const recipes = within(dialog).getByText("Recipes for this chat");
  fireEvent.click(recipes);
  fireEvent(recipes.parentElement!, new Event("toggle"));
  await waitFor(() => expect(within(dialog).getByRole("group", { name: "Use-case recipes" })).toBeVisible());
  await waitFor(() => expect(within(dialog).getByRole("combobox", { name: "Image generation recipe" })).toHaveValue("inherit"));
  const web = within(dialog).getByText("Web access");
  fireEvent.click(web);
  fireEvent(web.parentElement!, new Event("toggle"));
  const search = within(dialog).getByLabelText("Allow web searches");
  await waitFor(() => expect(search).toBeEnabled());
  fireEvent.click(search);
  await waitFor(() => expect(search).toBeChecked());
  const automatic = within(dialog).getByLabelText("Allow searches without asking again");
  await waitFor(() => expect(automatic).toBeEnabled());
  fireEvent.click(automatic);
  await waitFor(() => expect(automatic).toBeChecked());
  expect(api.updateChat).toHaveBeenLastCalledWith(chat.id, { web_settings_json: {
    allow_url_fetch: false, allow_search: true, allow_search_without_asking: true } });
  expect(title).toHaveValue("Unsaved new title");
  fireEvent.click(within(dialog).getByRole("button", { name: "Save chat" }));
  expect(save).toHaveBeenCalledWith(expect.objectContaining({ title: "Unsaved new title" }));
  client.clear();
});
