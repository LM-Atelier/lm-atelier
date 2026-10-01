import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { ChatManager } from "./ChatManager";
import { ChatSidebar } from "./ChatSidebar";
import { ProjectArchives } from "./ProjectArchives";
import type { Chat, ChatSummary, Project } from "./types";

vi.mock("./api", () => ({ api: {
  projects: vi.fn(), project: vi.fn(), chatSummaries: vi.fn(),
  exportProject: vi.fn(), importProject: vi.fn(),
} }));

const stamp = "2026-09-30T00:00:00Z";
const projects: Project[] = Array.from({ length: 75 }, (_, index) => ({
  id: `project-${index}`, name: `Notebook ${index}`, description: "", instructions: "",
  archived: index === 74, pinned: false, created_at: stamp, updated_at: stamp,
  image_workflow_revision_id: null, video_workflow_revision_id: null,
}));
const chat: Chat = {
  id: "chat-1", title: "Distant project chat", project_id: projects[74].id,
  archived: false, pinned: false, routing_mode: "auto", confirm_uncertain_media: true,
  active_chat_profile_id: null, active_image_profile_id: null, active_video_profile_id: null,
  active_head_message_id: null, created_at: stamp, updated_at: stamp,
};
const clients: QueryClient[] = [];
function show(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  render(<QueryClientProvider client={client}>{children}</QueryClientProvider>);
}
function sidebar() {
  return <ChatSidebar engines={[]} currentChatId={null} view="chat"
    onChat={vi.fn()} onSetup={vi.fn()} onView={vi.fn()} onNewChat={vi.fn()} onNewProject={vi.fn()}
    onExportProject={vi.fn()} onImportProject={vi.fn()} onUpdateChat={vi.fn()} onDeleteChat={vi.fn()}
    onUpdateProject={vi.fn()} onDeleteProject={vi.fn()}
    sidebar={{ width: 272, collapsed: false, setWidth: vi.fn(), toggle: vi.fn() }} />;
}
beforeEach(() => {
  vi.mocked(api.projects).mockImplementation(async (archived, query, options) => {
    const filtered = projects.filter((item) => (archived || !item.archived)
      && (!query || item.name.toLowerCase().includes(query.toLowerCase()))
      && (!options?.projectIds || options.projectIds.includes(item.id)));
    return filtered.slice(options?.offset ?? 0, (options?.offset ?? 0) + (options?.limit ?? filtered.length));
  });
  vi.mocked(api.project).mockImplementation(async (id) => projects.find((item) => item.id === id)!);
  vi.mocked(api.chatSummaries).mockResolvedValue([]);
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); vi.resetAllMocks(); });

it("pages archived project exports and searches projects beyond the loaded page", async () => {
  show(<ProjectArchives />);
  await screen.findByText("Notebook 49");
  expect(screen.queryByText("Notebook 74")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Load more projects" }));
  await screen.findByText("Notebook 74");
  fireEvent.change(screen.getByRole("searchbox", { name: "Search projects to export" }), { target: { value: "Notebook 74" } });
  await waitFor(() => expect(api.projects).toHaveBeenLastCalledWith(true, "Notebook 74", expect.objectContaining({ limit: 50, offset: 0, literalSearch: true })));
  expect(await screen.findByRole("button", { name: "Export Notebook 74 with media" })).toBeVisible();
});

it("keeps exported rows available and retries a failed next page", async () => {
  vi.mocked(api.projects).mockResolvedValueOnce(projects.slice(0, 50))
    .mockRejectedValueOnce(new Error("Page unavailable")).mockResolvedValueOnce(projects.slice(50));
  show(<ProjectArchives />);
  await screen.findByText("Notebook 49");
  fireEvent.click(screen.getByRole("button", { name: "Load more projects" }));
  await screen.findByText("Page unavailable");
  expect(screen.getByText("Notebook 0")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Retry projects" }));
  await screen.findByText("Notebook 74");
  expect(vi.mocked(api.projects).mock.calls.map((call) => call[2]?.offset)).toEqual([0, 50, 50]);
});

it("keeps an archived selected project outside the page while searching new choices", async () => {
  const save = vi.fn();
  show(<ChatManager chat={chat} onSave={save} onClose={vi.fn()} onDelete={vi.fn()} />);
  await screen.findByRole("option", { name: "Notebook 74 (Archived)" });
  expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue("project-74");
  fireEvent.change(screen.getByRole("searchbox", { name: "Search projects" }), { target: { value: "Notebook 63" } });
  await screen.findByRole("option", { name: "Notebook 63" });
  expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue("project-74");
  fireEvent.change(screen.getByRole("combobox", { name: "Project" }), { target: { value: "project-63" } });
  fireEvent.click(screen.getByRole("button", { name: "Save chat" }));
  expect(save).toHaveBeenCalledWith(expect.objectContaining({ project_id: "project-63" }));
});

it("loads exact off-page parents so their chats remain visible", async () => {
  const summary: ChatSummary = { ...chat, project_id: projects[63].id,
    activity: { active_work_count: 0, unresolved_failed_count: 0, last_output: null, last_failure: null } };
  vi.mocked(api.chatSummaries).mockResolvedValue([summary]);
  const original = vi.mocked(api.projects).getMockImplementation()!;
  vi.mocked(api.projects).mockImplementation(async (...args) => {
    const rows = await original(...args);
    // The parent was pinned after the first page was read.
    return args[2]?.projectIds ? rows.map((project) => ({ ...project, pinned: true })) : rows;
  });
  show(sidebar());
  await waitFor(() => expect(screen.getByRole("button", { name: "Distant project chat" })).toBeVisible());
  expect(api.projects).toHaveBeenCalledWith(true, "", expect.objectContaining({ projectIds: [projects[63].id], limit: 200 }));
  await waitFor(() => expect(document.querySelector(".project-main")).toHaveTextContent("Notebook 63"));
  expect(screen.queryByRole("button", { name: "Notebook 62" })).not.toBeInTheDocument();
});

it("keeps chat links visible when their parent lookup fails and allows retry", async () => {
  const summary: ChatSummary = { ...chat, project_id: projects[63].id,
    activity: { active_work_count: 0, unresolved_failed_count: 0, last_output: null, last_failure: null } };
  vi.mocked(api.chatSummaries).mockResolvedValue([summary]);
  const original = vi.mocked(api.projects).getMockImplementation()!;
  let fail = true;
  vi.mocked(api.projects).mockImplementation(async (...args) => {
    if (args[2]?.projectIds && fail) throw new Error("Parents unavailable");
    return original(...args);
  });
  show(sidebar());
  await screen.findByText("Parents unavailable");
  expect(screen.getByRole("button", { name: "Distant project chat" })).toBeVisible();
  fireEvent.change(screen.getByLabelText("Search projects and chats"), { target: { value: "Notebook" } });
  await waitFor(() => expect(api.chatSummaries).toHaveBeenLastCalledWith(null, false, "Notebook", expect.anything()));
  await waitFor(() => expect(screen.getByRole("button", { name: "Distant project chat" })).toBeVisible());
  fail = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry project names" }));
  expect(await screen.findByRole("button", { name: "Notebook 63" })).toBeVisible();
});

it("keeps server-matched project names when Unicode search differs from lowercase matching", async () => {
  vi.mocked(api.projects).mockImplementation(async (_archived, query) => query === "STRASSE"
    ? [{ ...projects[63], name: "Straße notebook" }] : []);
  show(sidebar());
  fireEvent.change(screen.getByLabelText("Search projects and chats"), { target: { value: "STRASSE" } });
  await waitFor(() => expect(api.projects).toHaveBeenLastCalledWith(false, "STRASSE",
    expect.objectContaining({ limit: 50, offset: 0, literalSearch: true })));
  expect(await screen.findByRole("button", { name: "Straße notebook" })).toBeVisible();
});
