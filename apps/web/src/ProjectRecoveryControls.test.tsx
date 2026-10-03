import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import { useState, type ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";
import { ProjectDeletionNotice } from "./ProjectDeletionNotice";
import { ProjectManager } from "./ProjectManager";
import { recoveryImpact, recoveryItem } from "./test/recoveryFixtures";
import { useProjectMutations } from "./useProjectMutations";
import type { Project } from "./types";

vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ApiError: actual.ApiError, api: { projectDeletionImpact: vi.fn(), trashProject: vi.fn(),
    recoveryImpact: vi.fn(), restoreRecovery: vi.fn(), deleteProject: vi.fn() } };
});
vi.mock("./PagedGenerationSettingsPanel", () => ({ PagedGenerationSettingsPanel: () => null }));
vi.mock("./WorkflowSelector", () => ({ WorkflowSelector: () => null }));
const project: Project = { id: "project-garden", name: "Garden layout", description: "", instructions: "",
  pinned: false, archived: false, image_workflow_revision_id: null, video_workflow_revision_id: null,
  created_at: "2026-10-02T00:00:00Z", updated_at: "2026-10-02T00:00:00Z" };
const item = { ...recoveryItem(project.id), kind: "project" as const, display_label: project.name };
const preview = { ...recoveryImpact(project.id), kind: "project" as const, counts: { ...item.counts, chats: 3 } };
const clients: QueryClient[] = [];
function setup() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  return { client, wrapper };
}
function Harness({ client }: { client: QueryClient }) {
  const [managed, setManaged] = useState(true);
  const { deleteProject } = useProjectMutations({ client });
  return <>
    {managed && <ProjectManager project={project} engines={[]} onClose={() => setManaged(false)} onSave={vi.fn()}
      onExport={vi.fn()} onDelete={async (command) => { await deleteProject.mutateAsync({ id: project.id, command }); setManaged(false); }} />}
    <ProjectDeletionNotice deletion={deleteProject} />
  </>;
}
function show() {
  const { client, wrapper } = setup();
  render(<Harness client={client} />, { wrapper });
  return client;
}
async function confirmTrash() {
  fireEvent.click(screen.getByRole("button", { name: "Delete project" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Move to Recently Deleted" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.projectDeletionImpact).mockResolvedValue(preview);
  vi.mocked(api.trashProject).mockResolvedValue(item);
  vi.mocked(api.recoveryImpact).mockResolvedValue({ ...preview, available_actions: ["restore", "purge"] });
  vi.mocked(api.restoreRecovery).mockResolvedValue({ kind: "project", subject_id: project.id, deletion_id: item.deletion_id,
    action: "restore", replayed: false, reclaimed_bytes: 0 });
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); vi.useRealTimers(); });

it("waits for a fresh Project impact and keeps Cancel inert", async () => {
  let resolve!: (value: typeof preview) => void;
  vi.mocked(api.projectDeletionImpact).mockReturnValue(new Promise((finish) => { resolve = finish; }));
  show();
  fireEvent.click(screen.getByRole("button", { name: "Delete project" }));
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  expect(api.trashProject).not.toHaveBeenCalled();
  resolve(preview);
  const dialog = screen.getByRole("dialog", { name: "Move this project to Recently Deleted?" });
  expect(await within(dialog).findByText(/3 chats stay available/)).toBeVisible();
  fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
  expect(api.trashProject).not.toHaveBeenCalled();
  expect(api.deleteProject).not.toHaveBeenCalled();
  expect(screen.getByRole("dialog", { name: "Manage project" })).toBeVisible();
});

it("moves only the Project and focuses Undo while preserving loaded child history", async () => {
  const client = show();
  const history = { id: "chat-garden", messages: [{ id: "garden-message", text: "Keep the paths clear" }] };
  client.setQueryData(["chat", history.id], history);
  const invalidate = vi.spyOn(client, "invalidateQueries");
  await confirmTrash();
  expect(await screen.findByRole("button", { name: "Undo" })).toHaveFocus();
  expect(screen.queryByRole("dialog", { name: "Manage project" })).toBeNull();
  expect(api.trashProject).toHaveBeenCalledWith(project.id, {
    expected_revision: preview.revision, impact_sha256: preview.impact_sha256, operation_key: expect.any(String),
  });
  expect(api.deleteProject).not.toHaveBeenCalled();
  expect(client.getQueryData(["chat", history.id])).toEqual(history);
  fireEvent.click(screen.getByRole("button", { name: "Undo" }));
  await waitFor(() => expect(screen.queryByRole("button", { name: "Undo" })).toBeNull());
  expect(api.restoreRecovery).toHaveBeenCalledWith(item.deletion_id, expect.objectContaining({ restore_unfiled: false }));
  expect(client.getQueryData(["chat", history.id])).toEqual(history);
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["projects"] });
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["chat"] });
});

it("keeps the manager open after stale consent and requires a fresh confirmation", async () => {
  vi.mocked(api.trashProject).mockRejectedValueOnce(new ApiError(409, undefined, "The project changed.", "recovery-impact-stale"));
  show();
  await confirmTrash();
  await screen.findByText("The project changed.");
  const first = vi.mocked(api.trashProject).mock.calls[0][1];
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  expect(api.trashProject).toHaveBeenCalledTimes(1);
  vi.mocked(api.projectDeletionImpact).mockResolvedValue({ ...preview, revision: "d".repeat(64), counts: { ...preview.counts, chats: 4 } });
  fireEvent.click(screen.getByRole("button", { name: "Check again" }));
  await screen.findByText(/4 chats stay available/);
  expect(api.trashProject).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  await screen.findByRole("button", { name: "Undo" });
  const second = vi.mocked(api.trashProject).mock.calls[1][1];
  expect(second.expected_revision).toBe("d".repeat(64));
  expect(second.operation_key).not.toBe(first.operation_key);
});

it("retries an uncertain Trash result with exactly the same command", async () => {
  vi.mocked(api.trashProject).mockRejectedValueOnce(new Error("Connection interrupted."));
  show();
  await confirmTrash();
  fireEvent.click(await screen.findByRole("button", { name: "Try again" }));
  await screen.findByRole("button", { name: "Undo" });
  expect(vi.mocked(api.trashProject).mock.calls[1]).toEqual(vi.mocked(api.trashProject).mock.calls[0]);
  expect(api.projectDeletionImpact).toHaveBeenCalledTimes(1);
});

it("retries an uncertain Undo with identical consent and rechecks a stale Undo explicitly", async () => {
  vi.mocked(api.restoreRecovery).mockRejectedValueOnce(new Error("Connection interrupted."))
    .mockRejectedValueOnce(new ApiError(409, undefined, "The filing changed.", "recovery-impact-stale"));
  show();
  await confirmTrash();
  fireEvent.click(await screen.findByRole("button", { name: "Undo" }));
  fireEvent.click(await screen.findByRole("button", { name: "Try Undo again" }));
  await screen.findByText("The filing changed.");
  expect(vi.mocked(api.restoreRecovery).mock.calls[1]).toEqual(vi.mocked(api.restoreRecovery).mock.calls[0]);
  expect(api.recoveryImpact).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "Recheck Undo" }));
  expect(api.restoreRecovery).toHaveBeenCalledTimes(2);
  fireEvent.click(await screen.findByRole("button", { name: "Undo" }));
  await waitFor(() => expect(api.restoreRecovery).toHaveBeenCalledTimes(3));
  expect(api.recoveryImpact).toHaveBeenCalledTimes(2);
  expect(vi.mocked(api.restoreRecovery).mock.calls[2][1].operation_key).not.toBe(vi.mocked(api.restoreRecovery).mock.calls[0][1].operation_key);
});

it("does not dismiss a newer deleted Project when an earlier Undo finishes", async () => {
  const { client, wrapper } = setup();
  let finish!: (value: Awaited<ReturnType<typeof api.restoreRecovery>>) => void;
  vi.mocked(api.restoreRecovery).mockReturnValueOnce(new Promise((resolve) => { finish = resolve; }));
  const { result } = renderHook(() => useProjectMutations({ client }), { wrapper });
  const command = { expected_revision: preview.revision, impact_sha256: preview.impact_sha256, operation_key: "trash-garden" };
  await act(async () => { await result.current.deleteProject.mutateAsync({ id: project.id, command }); });
  act(() => result.current.deleteProject.undo.mutate(item));
  await waitFor(() => expect(api.restoreRecovery).toHaveBeenCalledTimes(1));
  const next = { ...item, deletion_id: "deleted-harbor", subject_id: "project-harbor", display_label: "Harbor layout" };
  vi.mocked(api.trashProject).mockResolvedValueOnce(next);
  await act(async () => { await result.current.deleteProject.mutateAsync({ id: next.subject_id, command }); });
  await act(async () => { finish({ kind: "project", subject_id: item.subject_id, deletion_id: item.deletion_id,
    action: "restore", replayed: false, reclaimed_bytes: 0 }); });
  expect(result.current.deleteProject.deleted).toEqual(next);
});

it("closes Project Undo at the exact original deadline without refreshing", async () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-10-02T00:00:00Z"));
  const deadline = Date.now() + 60_000;
  vi.mocked(api.trashProject).mockResolvedValue({ ...item, purge_after: new Date(deadline).toISOString() });
  show();
  fireEvent.click(screen.getByRole("button", { name: "Delete project" }));
  await act(async () => { await vi.advanceTimersByTimeAsync(20); });
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  await act(async () => { await vi.advanceTimersByTimeAsync(20); });
  const undo = screen.getByRole("button", { name: "Undo" });
  expect(undo).toHaveAttribute("aria-disabled", "false");
  await act(async () => { await vi.advanceTimersByTimeAsync(deadline - Date.now() - 1); });
  expect(undo).toHaveAttribute("aria-disabled", "false");
  await act(async () => { await vi.advanceTimersByTimeAsync(1); });
  expect(undo).toHaveAttribute("aria-disabled", "true");
  expect(screen.getByText(/Recovery ended/)).toBeVisible();
  fireEvent.click(undo);
  expect(api.restoreRecovery).not.toHaveBeenCalled();
});
