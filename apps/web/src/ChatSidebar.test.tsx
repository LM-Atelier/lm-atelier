import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ChatSidebar } from "./ChatSidebar";
import type { Chat, Project } from "./types";
import { changeChatPages, restoreChatPages, snapshotChatPages, useChatPages } from "./useChatPages";
import { useProjectMutations } from "./useProjectMutations";

vi.mock("./api", () => ({ api: { chats: vi.fn(), importProject: vi.fn() } }));

const stamp = "2026-09-01T00:00:00Z";
function chat(number: number): Chat {
  return {
    id: `chat-${number}`, title: `Notebook ${number}`, project_id: null, archived: false,
    pinned: false, routing_mode: "auto", confirm_uncertain_media: false,
    active_chat_profile_id: null, active_image_profile_id: null, active_video_profile_id: null,
    active_head_message_id: null, created_at: stamp, updated_at: stamp,
  };
}
const rows = Array.from({ length: 125 }, (_, index) => chat(index));
const project: Project = {
  id: "project-imported", name: "Harbor notebook", description: "", instructions: "",
  pinned: false, archived: false, image_workflow_revision_id: null, video_workflow_revision_id: null,
  created_at: stamp, updated_at: stamp,
};

function setup() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  return { client, wrapper };
}

function sidebar(onChat = vi.fn()) {
  return <ChatSidebar projects={[project]} engines={[]} presets={[]} currentChatId={null} view="chat"
    onChat={onChat} onSetup={vi.fn()} onView={vi.fn()} onNewChat={vi.fn()} onNewProject={vi.fn()}
    onExportProject={vi.fn()} onImportProject={vi.fn()} onUpdateChat={vi.fn()} onDeleteChat={vi.fn()}
    onUpdateProject={vi.fn()} onDeleteProject={vi.fn()}
    sidebar={{ width: 272, collapsed: false, setWidth: vi.fn(), toggle: vi.fn() }} />;
}

beforeEach(() => {
  vi.mocked(api.chats).mockReset();
  vi.mocked(api.chats).mockImplementation(async (_project, _archived, _query, options) => {
    expect(options?.limit).toBe(50);
    const offset = options?.offset ?? 0;
    return rows.slice(offset, offset + 50);
  });
});
afterEach(cleanup);

it("loads a bounded first page and opens chats on later pages", async () => {
  const { wrapper } = setup();
  const onChat = vi.fn();
  render(sidebar(onChat), { wrapper });
  expect(await screen.findByRole("button", { name: "Notebook 49" })).toBeVisible();
  expect(screen.queryByRole("button", { name: "Notebook 50" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Load more chats" }));
  fireEvent.click(await screen.findByRole("button", { name: "Notebook 99" }));
  expect(onChat).toHaveBeenCalledWith("chat-99");
  fireEvent.click(screen.getByRole("button", { name: "Load more chats" }));
  expect(await screen.findByRole("button", { name: "Notebook 124" })).toBeVisible();
  expect(screen.queryByRole("button", { name: "Load more chats" })).not.toBeInTheDocument();
  expect(vi.mocked(api.chats).mock.calls.map((call) => call[3]?.offset)).toEqual([0, 50, 100]);
});

it("searches unloaded chats and project names and restarts paging for archive changes", async () => {
  const { wrapper } = setup();
  vi.mocked(api.chats).mockImplementation(async (_project, archived = false, query, options) => {
    expect(options).toMatchObject({ offset: 0, limit: 50, searchProjects: true });
    if (query === "Harbor") return [{ ...chat(124), project_id: project.id, archived }];
    if (query === "Notebook 124") return [chat(124)];
    return rows.slice(0, 50);
  });
  render(sidebar(), { wrapper });
  await screen.findByRole("button", { name: "Notebook 0" });
  fireEvent.change(screen.getByLabelText("Search projects and chats"), { target: { value: "Notebook 124" } });
  expect(await screen.findByRole("button", { name: "Notebook 124" })).toBeVisible();
  fireEvent.change(screen.getByLabelText("Search projects and chats"), { target: { value: "Harbor" } });
  expect(await screen.findByRole("button", { name: "Notebook 124" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Archived" }));
  expect(await screen.findByRole("button", { name: "Notebook 124 Archived" })).toBeVisible();
  expect(api.chats).toHaveBeenLastCalledWith(null, true, "Harbor", expect.objectContaining({ limit: 50, offset: 0 }));
});

it("keeps loaded rows and retries a failed next page at its original offset", async () => {
  const { wrapper } = setup();
  vi.mocked(api.chats).mockResolvedValueOnce(rows.slice(0, 50)).mockRejectedValueOnce(new Error("offline"))
    .mockResolvedValueOnce(rows.slice(50, 75));
  render(sidebar(), { wrapper });
  await screen.findByRole("button", { name: "Notebook 0" });
  fireEvent.click(screen.getByRole("button", { name: "Load more chats" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not load chats");
  expect(screen.getByRole("button", { name: "Notebook 0" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByRole("button", { name: "Notebook 74" })).toBeVisible();
  expect(vi.mocked(api.chats).mock.calls.map((call) => call[3]?.offset)).toEqual([0, 50, 50]);
});

it("updates and rolls back every loaded page without shifting pagination offsets", async () => {
  const { client, wrapper } = setup();
  const { result } = renderHook(() => useChatPages(), { wrapper });
  await waitFor(() => expect(result.current.data).toHaveLength(50));
  await act(async () => { await result.current.fetchNextPage(); });
  const snapshot = snapshotChatPages(client);
  act(() => changeChatPages(client, (item) => item.id === "chat-75" ? { ...item, title: "Changed" } : item));
  await waitFor(() => expect(result.current.data?.find((item) => item.id === "chat-75")?.title).toBe("Changed"));
  act(() => changeChatPages(client, (item) => item.id === "chat-0" ? null : item));
  await waitFor(() => expect(result.current.data).toHaveLength(99));
  act(() => restoreChatPages(client, snapshot));
  await waitFor(() => expect(result.current.data).toHaveLength(100));
  expect(result.current.data?.find((item) => item.id === "chat-75")?.title).toBe("Notebook 75");
  await act(async () => { await result.current.fetchNextPage(); });
  expect(vi.mocked(api.chats).mock.calls.at(-1)?.[3]?.offset).toBe(100);
});

it("opens an imported project's chat with a bounded lookup even when no loaded page contains it", async () => {
  const { client, wrapper } = setup();
  vi.mocked(api.importProject).mockResolvedValue(project);
  vi.mocked(api.chats).mockResolvedValue([{ ...chat(124), project_id: project.id }]);
  const onImportedChat = vi.fn();
  const { result } = renderHook(() => useProjectMutations({ client, onImportedChat }), { wrapper });
  await act(async () => { await result.current.importProject.mutateAsync(new File(["archive"], "project.zip")); });
  expect(api.chats).toHaveBeenCalledWith(project.id, true, "", { limit: 1, offset: 0 });
  expect(onImportedChat).toHaveBeenCalledWith("chat-124");
});
