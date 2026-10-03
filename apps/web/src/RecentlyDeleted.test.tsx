import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";
import { RecentlyDeleted } from "./RecentlyDeleted";
import type { RecoveryImpact, RecoveryItem } from "./recoveryTypes";

vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ApiError: actual.ApiError, api: {
    recoveryItems: vi.fn(), recoveryImpact: vi.fn(), restoreRecovery: vi.fn(), purgeRecovery: vi.fn(),
  } };
});

const item: RecoveryItem = {
  deletion_id: "deleted-garden", kind: "chat", subject_id: "chat-garden", display_label: "Garden notes",
  original_location: { project_id: "project-garden", project_label: "Garden" },
  deleted_at: "2026-10-02T00:00:00Z", purge_after: "2099-11-01T00:00:00Z", state: "recoverable",
  revision: "a".repeat(64), restore_conflicts: [], delete_generated_media: false,
  counts: { chats: 0, messages: 4, message_parts: 4, response_revisions: 2, runs: 2, jobs: 2, work_plans: 2,
    workflow_families: 0, workflow_definitions: 0, workflow_revisions: 0,
    references: 1, artifacts: 3, active_work: 0, retained_bytes: 1024, reclaimable_bytes: 0 },
};
const preview: RecoveryImpact = {
  kind: "chat", subject_id: item.subject_id, revision: "b".repeat(64), impact_sha256: "c".repeat(64),
  counts: item.counts, conflicts: [], available_actions: ["restore", "purge"],
  delete_generated_media: false, reclaimed_bytes: 0,
};
const clients: QueryClient[] = [];

it("explains the fresh media purge intent before permanent chat deletion", async () => {
  vi.mocked(api.recoveryImpact).mockResolvedValue({ ...preview, delete_generated_media: true });
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Permanently delete Garden notes" }));
  await screen.findByText(/This chat's saved choice also removes exclusive generated media from the Media Library/);
  expect(screen.getByText(/Favorites and media retained elsewhere stay protected/)).toBeTruthy();
  expect(api.purgeRecovery).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(api.purgeRecovery).not.toHaveBeenCalled();
});

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><RecentlyDeleted /></QueryClientProvider>);
  return client;
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.recoveryItems).mockResolvedValue({ items: [item], next_cursor: null });
  vi.mocked(api.recoveryImpact).mockResolvedValue(preview);
  vi.mocked(api.restoreRecovery).mockResolvedValue({ deletion_id: item.deletion_id, kind: "chat",
    subject_id: item.subject_id, action: "restore", replayed: false, reclaimed_bytes: 0 });
  vi.mocked(api.purgeRecovery).mockResolvedValue({ deletion_id: item.deletion_id, kind: "chat",
    subject_id: item.subject_id, action: "purge", replayed: false, reclaimed_bytes: 0 });
});
afterEach(() => { cleanup(); for (const client of clients.splice(0)) client.clear(); vi.useRealTimers(); });

it("loads another bounded page while showing exact expiry and structural impact", async () => {
  vi.mocked(api.recoveryItems).mockResolvedValueOnce({ items: [item], next_cursor: "page-two" })
    .mockResolvedValue({ items: [{ ...item, deletion_id: "deleted-travel", display_label: "Travel notes" }], next_cursor: null });
  show();
  await screen.findByText("Garden notes");
  const row = screen.getByText("Garden notes").closest("article")!;
  expect(row.querySelector('time[datetime="2099-11-01T00:00:00Z"]')).toBeTruthy();
  expect(within(row).getByText(/4 messages · 3 media files · 2 generations/)).toBeTruthy();
  expect(screen.queryByText(item.revision)).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Load more deleted items" }));
  await screen.findByText("Travel notes");
  expect(screen.getByText("Garden notes")).toBeTruthy();
  expect(api.recoveryItems).toHaveBeenLastCalledWith(expect.objectContaining({ cursor: "page-two" }));
});

it("waits for a fresh impact and requires an explicit unfiled restore choice", async () => {
  let finish!: (value: RecoveryImpact) => void;
  vi.mocked(api.recoveryImpact).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Restore Garden notes" }));
  fireEvent.click(screen.getByRole("button", { name: "Restore chat" }));
  expect(api.restoreRecovery).not.toHaveBeenCalled();
  finish({ ...preview, conflicts: ["original_project_missing"] });
  const choice = await screen.findByRole("checkbox", { name: /Restore this chat unfiled/ });
  fireEvent.click(screen.getByRole("button", { name: "Restore chat" }));
  expect(api.restoreRecovery).not.toHaveBeenCalled();
  fireEvent.click(choice);
  fireEvent.click(screen.getByRole("button", { name: "Restore chat" }));
  await waitFor(() => expect(api.restoreRecovery).toHaveBeenCalledWith(item.deletion_id, {
    expected_revision: preview.revision, impact_sha256: preview.impact_sha256,
    operation_key: expect.any(String), restore_unfiled: true,
  }));
  expect(await screen.findByText("Chat restored with its original history.")).toBeTruthy();
});

it("confirms permanent deletion separately and sends the current impact and acknowledgement", async () => {
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Permanently delete Garden notes" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Delete permanently" })).toHaveAttribute("aria-disabled", "false"));
  expect(api.purgeRecovery).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(api.purgeRecovery).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Permanently delete Garden notes" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Delete permanently" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(screen.getByRole("button", { name: "Delete permanently" }));
  await waitFor(() => expect(api.purgeRecovery).toHaveBeenCalledWith(item.deletion_id, {
    expected_revision: preview.revision, impact_sha256: preview.impact_sha256,
    operation_key: expect.any(String), acknowledgement: "permanently-delete",
  }));
});

it("keeps a failed preview inert until the user checks again", async () => {
  vi.mocked(api.recoveryImpact).mockRejectedValueOnce(new Error("The database is busy."));
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Restore Garden notes" }));
  await screen.findByText("The database is busy.");
  fireEvent.click(screen.getByRole("button", { name: "Restore chat" }));
  expect(api.restoreRecovery).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Recheck recovery details" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Restore chat" })).toHaveAttribute("aria-disabled", "false"));
  expect(api.restoreRecovery).not.toHaveBeenCalled();
});

it("requires another confirmation after a stale impact refusal", async () => {
  vi.mocked(api.purgeRecovery).mockRejectedValueOnce(new ApiError(409, undefined, "The conversation changed.", "recovery-impact-stale"));
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Permanently delete Garden notes" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Delete permanently" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(screen.getByRole("button", { name: "Delete permanently" }));
  await screen.findByText("The conversation changed.");
  vi.mocked(api.recoveryImpact).mockResolvedValue({ ...preview, revision: "d".repeat(64), counts: { ...item.counts, messages: 5 } });
  fireEvent.click(screen.getByRole("button", { name: "Recheck recovery details" }));
  await screen.findByText(/5 messages · 3 media files/);
  expect(api.purgeRecovery).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "Delete permanently" }));
  await waitFor(() => expect(api.purgeRecovery).toHaveBeenCalledTimes(2));
  expect(vi.mocked(api.purgeRecovery).mock.calls[1]?.[1].expected_revision).toBe("d".repeat(64));
});

it("retries an uncertain result with the identical command rather than a new preview", async () => {
  vi.mocked(api.restoreRecovery).mockRejectedValueOnce(new Error("Connection interrupted."));
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Restore Garden notes" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Restore chat" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(screen.getByRole("button", { name: "Restore chat" }));
  const retry = await screen.findByRole("button", { name: "Try again" });
  fireEvent.click(retry);
  await waitFor(() => expect(api.restoreRecovery).toHaveBeenCalledTimes(2));
  expect(vi.mocked(api.restoreRecovery).mock.calls[1]).toEqual(vi.mocked(api.restoreRecovery).mock.calls[0]);
  expect(api.recoveryImpact).toHaveBeenCalledTimes(1);
});

it("sends state and date filters to the server before paging", async () => {
  show();
  await screen.findByText("Garden notes");
  fireEvent.change(screen.getByLabelText("State"), { target: { value: "blocked" } });
  fireEvent.change(screen.getByLabelText("Deleted since (UTC)"), { target: { value: "2026-10-01" } });
  await waitFor(() => expect(api.recoveryItems).toHaveBeenLastCalledWith(expect.objectContaining({
    state: "blocked", deletedSince: "2026-10-01T00:00:00Z", cursor: undefined,
  })));
});

it("closes the restore window at its deadline without needing a page refresh", async () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-10-02T00:00:00Z"));
  const deadline = Date.now() + 60_000;
  vi.mocked(api.recoveryItems).mockResolvedValue({ items: [{ ...item, purge_after: new Date(deadline).toISOString() }], next_cursor: null });
  show();
  await act(async () => { await vi.advanceTimersByTimeAsync(20); });
  const restore = screen.getByRole("button", { name: "Restore Garden notes" });
  expect(restore).toHaveAttribute("aria-disabled", "false");
  await act(async () => { await vi.advanceTimersByTimeAsync(59_979); });
  expect(restore).toHaveAttribute("aria-disabled", "false");
  await act(async () => { await vi.advanceTimersByTimeAsync(1); });
  expect(restore).toHaveAttribute("aria-disabled", "true");
  expect(screen.getByText(/Recovery ended/)).toBeTruthy();
  fireEvent.click(restore);
  expect(api.recoveryImpact).not.toHaveBeenCalled();
});
