import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";
import type { RecoveryBatchPreview, RecoveryBatchResult } from "./recoveryTypes";
import { RecentlyDeleted } from "./RecentlyDeleted";
import { recoveryImpact, recoveryItem } from "./test/recoveryFixtures";

vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ApiError: actual.ApiError, api: {
    recoveryItems: vi.fn(), recoveryImpact: vi.fn(), restoreRecovery: vi.fn(), purgeRecovery: vi.fn(),
    previewRecoveryBatch: vi.fn(), applyRecoveryBatch: vi.fn(),
  } };
});

const clients: QueryClient[] = [];
const garden = recoveryItem("garden");
const travel = { ...recoveryItem("travel"), display_label: "Travel notes" };
const preview: RecoveryBatchPreview = {
  batch_id: "selection-garden-travel", revision: "b".repeat(64), impact_sha256: "c".repeat(64),
  action: "restore", policy: "all-or-nothing", restore_unfiled: false, available: true,
  expires_at: "2099-10-02T00:15:00Z",
  items: [garden, travel].map(item => ({ deletion_id: item.deletion_id, display_label: item.display_label,
    purge_after: item.purge_after, impact: recoveryImpact(item.subject_id, ["restore", "purge"]) })),
};
const result: RecoveryBatchResult = {
  batch_id: preview.batch_id, action: "restore", policy: "all-or-nothing", reclaimed_bytes: 0,
  results: [garden, travel].map(item => ({ deletion_id: item.deletion_id, kind: item.kind, subject_id: item.subject_id,
    action: "restore", replayed: false, reclaimed_bytes: 0 })),
};
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><RecentlyDeleted /></QueryClientProvider>);
  return client;
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.recoveryItems).mockResolvedValue({ items: [garden, travel], next_cursor: null });
  vi.mocked(api.previewRecoveryBatch).mockResolvedValue(preview);
  vi.mocked(api.applyRecoveryBatch).mockResolvedValue(result);
});
afterEach(() => { cleanup(); clients.splice(0).forEach(client => client.clear()); vi.useRealTimers(); });

async function selectBoth() {
  fireEvent.click(await screen.findByRole("checkbox", { name: "Select Garden notes" }));
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Travel notes" }));
}
async function review(action: "restore" | "purge" = "restore") {
  fireEvent.click(screen.getByRole("button", { name: action === "restore" ? "Review restore selection" : "Review permanent deletion" }));
  const dialog = screen.getByRole("dialog");
  await waitFor(() => expect(within(dialog).queryByText("Checking the whole selection…")).toBeNull());
  return dialog;
}

it("offers a bounded multi-item selection without submitting individual recovery commands", async () => {
  show();
  fireEvent.click(await screen.findByRole("checkbox", { name: "Select Garden notes" }));
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Travel notes" }));
  expect(screen.getByText("2 of 20 items selected")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Review restore selection" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Review permanent deletion" })).toBeTruthy();
  expect(api.restoreRecovery).not.toHaveBeenCalled();
  expect(api.purgeRecovery).not.toHaveBeenCalled();
});

it("waits for one materialized preview and sends one atomic command for the whole selection", async () => {
  let finish!: (value: RecoveryBatchPreview) => void;
  vi.mocked(api.previewRecoveryBatch).mockReturnValue(new Promise(resolve => { finish = resolve; }));
  const client = show(); const invalidate = vi.spyOn(client, "invalidateQueries");
  await selectBoth();
  fireEvent.click(screen.getByRole("button", { name: "Review restore selection" }));
  const dialog = screen.getByRole("dialog", { name: "Restore selected items?" });
  expect(within(dialog).getByText(/All or nothing: every selected item/)).toBeTruthy();
  fireEvent.click(within(dialog).getByRole("button", { name: "Restore selected items" }));
  expect(api.applyRecoveryBatch).not.toHaveBeenCalled();
  await act(async () => { finish(preview); });
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Restore selected items" })).toHaveAttribute("aria-disabled", "false"));
  expect(api.previewRecoveryBatch).toHaveBeenCalledTimes(1);
  expect(api.previewRecoveryBatch).toHaveBeenCalledWith({ deletion_ids: [garden.deletion_id, travel.deletion_id], action: "restore", restore_unfiled: false }, expect.any(AbortSignal));
  expect(dialog.querySelector(`time[datetime="${garden.purge_after}"]`)).toBeTruthy();
  expect(dialog.querySelector(`time[datetime="${preview.expires_at}"]`)).toBeTruthy();
  fireEvent.click(within(dialog).getByRole("button", { name: "Restore selected items" }));
  await waitFor(() => expect(api.applyRecoveryBatch).toHaveBeenCalledWith(preview.batch_id, {
    expected_revision: preview.revision, impact_sha256: preview.impact_sha256, operation_key: expect.any(String),
  }));
  const notice = await screen.findByText(/2 items restored together/);
  expect(notice).toHaveFocus();
  expect(screen.queryByRole("dialog")).toBeNull();
  for (const key of ["recovery-items", "projects", "artifact-library-v1", "workflow-families"]) expect(invalidate).toHaveBeenCalledWith({ queryKey: [key] });
  expect(api.restoreRecovery).not.toHaveBeenCalled(); expect(api.purgeRecovery).not.toHaveBeenCalled();
});

it("requires permanent deletion acknowledgement and explains overlapping storage estimates", async () => {
  vi.mocked(api.previewRecoveryBatch).mockResolvedValue({ ...preview, action: "purge", items: preview.items.map(item => ({ ...item, impact: { ...item.impact, delete_generated_media: true } })) });
  vi.mocked(api.applyRecoveryBatch).mockResolvedValue({ ...result, action: "purge" });
  show(); await selectBoth(); const dialog = await review("purge");
  expect(within(dialog).getByText(/Per-item storage estimates may overlap/)).toBeTruthy();
  expect(within(dialog).getAllByText(/Favorites and media retained elsewhere stay protected/)).toHaveLength(2);
  const button = within(dialog).getByRole("button", { name: "Delete selected permanently" });
  expect(button).toHaveAttribute("aria-disabled", "true"); fireEvent.click(button);
  expect(api.applyRecoveryBatch).not.toHaveBeenCalled();
  fireEvent.click(within(dialog).getByRole("checkbox", { name: /permanent deletion cannot be undone/ }));
  fireEvent.click(button);
  await waitFor(() => expect(api.applyRecoveryBatch).toHaveBeenCalledWith(preview.batch_id, expect.objectContaining({ acknowledgement: "permanently-delete" })));
  expect(await screen.findByText(/2 items permanently deleted together/)).toBeTruthy();
});

it("refuses a whole selection when any member is unavailable and preserves the per-item impact", async () => {
  vi.mocked(api.previewRecoveryBatch).mockResolvedValue({ ...preview, available: false, items: preview.items.map((item, index) => index === 1
    ? { ...item, impact: { ...item.impact, available_actions: [], counts: { ...item.impact.counts, messages: 14, active_work: 1 } } } : item) });
  show(); await selectBoth(); const dialog = await review();
  expect(within(dialog).getByText(/14 messages/)).toBeTruthy();
  expect(within(dialog).getByText(/None of the selected items will be changed/)).toBeTruthy();
  fireEvent.click(within(dialog).getByRole("button", { name: "Restore selected items" }));
  expect(api.applyRecoveryBatch).not.toHaveBeenCalled();
});

it("materializes a new preview for the explicit unfiled choice before allowing confirmation", async () => {
  const missing = { ...preview, available: false, items: preview.items.map(item => ({ ...item, impact: { ...item.impact, conflicts: ["original_project_missing" as const] } })) };
  const updated = { ...missing, available: true, restore_unfiled: true, batch_id: "unfiled-selection", revision: "d".repeat(64), impact_sha256: "e".repeat(64) };
  vi.mocked(api.previewRecoveryBatch).mockResolvedValueOnce(missing).mockResolvedValue(updated);
  show(); await selectBoth(); const dialog = await review();
  fireEvent.click(within(dialog).getByRole("button", { name: "Restore selected items" }));
  expect(api.applyRecoveryBatch).not.toHaveBeenCalled();
  fireEvent.click(within(dialog).getByRole("checkbox", { name: /original project is gone as unfiled/ }));
  await waitFor(() => expect(api.previewRecoveryBatch).toHaveBeenLastCalledWith(expect.objectContaining({ restore_unfiled: true }), expect.any(AbortSignal)));
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Restore selected items" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(within(dialog).getByRole("button", { name: "Restore selected items" }));
  await waitFor(() => expect(api.applyRecoveryBatch).toHaveBeenCalledWith(updated.batch_id, {
    expected_revision: updated.revision, impact_sha256: updated.impact_sha256, operation_key: expect.any(String),
  }));
});

it("requires explicit recheck and a new confirmation after a stale batch refusal", async () => {
  vi.mocked(api.applyRecoveryBatch).mockRejectedValueOnce(new ApiError(409, "Selection changed", "Selection changed", "recovery-impact-stale")).mockResolvedValue(result);
  const next = { ...preview, batch_id: "rechecked-selection", revision: "d".repeat(64), impact_sha256: "e".repeat(64) };
  vi.mocked(api.previewRecoveryBatch).mockResolvedValueOnce(preview).mockResolvedValue(next);
  show(); await selectBoth(); const dialog = await review();
  fireEvent.click(within(dialog).getByRole("button", { name: "Restore selected items" }));
  await within(dialog).findByText("Selection changed");
  const first = vi.mocked(api.applyRecoveryBatch).mock.calls[0][1];
  fireEvent.click(within(dialog).getByRole("button", { name: "Restore selected items" }));
  expect(api.applyRecoveryBatch).toHaveBeenCalledTimes(1); expect(api.previewRecoveryBatch).toHaveBeenCalledTimes(1);
  fireEvent.click(within(dialog).getByRole("button", { name: "Recheck selected recovery details" }));
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Restore selected items" })).toHaveAttribute("aria-disabled", "false"));
  expect(api.applyRecoveryBatch).toHaveBeenCalledTimes(1);
  fireEvent.click(within(dialog).getByRole("button", { name: "Restore selected items" }));
  await waitFor(() => expect(api.applyRecoveryBatch).toHaveBeenCalledTimes(2));
  const [id, second] = vi.mocked(api.applyRecoveryBatch).mock.calls[1];
  expect(id).toBe(next.batch_id); expect(second.expected_revision).toBe(next.revision);
  expect(second.operation_key).not.toBe(first.operation_key);
});

it("preserves an uncertain request across closing, list updates and reopening", async () => {
  vi.mocked(api.previewRecoveryBatch).mockResolvedValue({ ...preview, action: "purge" });
  vi.mocked(api.applyRecoveryBatch).mockRejectedValueOnce(new Error("Connection interrupted")).mockResolvedValue(result);
  const client = show(); await selectBoth(); let dialog = await review("purge");
  fireEvent.click(within(dialog).getByRole("checkbox", { name: /permanent deletion cannot be undone/ }));
  fireEvent.click(within(dialog).getByRole("button", { name: "Delete selected permanently" }));
  await within(dialog).findByText("Connection interrupted");
  const first = vi.mocked(api.applyRecoveryBatch).mock.calls[0];
  expect(within(dialog).queryByRole("button", { name: "Recheck selected recovery details" })).toBeNull();
  fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
  expect(screen.queryByRole("dialog")).toBeNull();
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Garden notes" }));
  expect(screen.getByText("2 of 20 items selected")).toBeTruthy();
  await act(async () => { await client.invalidateQueries({ queryKey: ["recovery-items"] }); });
  fireEvent.click(screen.getByRole("button", { name: "Review pending action" }));
  dialog = screen.getByRole("dialog");
  expect(within(dialog).getByRole("checkbox", { name: /permanent deletion cannot be undone/ })).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(within(dialog).getByRole("button", { name: "Try this request again" }));
  await waitFor(() => expect(api.applyRecoveryBatch).toHaveBeenCalledTimes(2));
  expect(vi.mocked(api.applyRecoveryBatch).mock.calls[1]).toEqual(first);
  expect(api.previewRecoveryBatch).toHaveBeenCalledTimes(1);
});

it("keeps an in-flight batch fixed against duplicate submit, Cancel, Escape and behind-dialog selection", async () => {
  let finish!: (value: RecoveryBatchResult) => void;
  vi.mocked(api.applyRecoveryBatch).mockReturnValue(new Promise(resolve => { finish = resolve; }));
  show(); await selectBoth(); const dialog = await review();
  const confirm = within(dialog).getByRole("button", { name: "Restore selected items" });
  fireEvent.click(confirm); fireEvent.click(confirm);
  fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
  fireEvent.keyDown(dialog, { key: "Escape" });
  fireEvent.change(screen.getByLabelText("Type"), { target: { value: "project" } });
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Garden notes" }));
  await waitFor(() => expect(api.applyRecoveryBatch).toHaveBeenCalledTimes(1));
  expect(screen.getByRole("dialog")).toBe(dialog);
  expect(screen.getByText("2 of 20 items selected")).toBeTruthy();
  expect(screen.getByLabelText("Type")).toHaveValue("");
  await act(async () => { finish(result); });
  expect(await screen.findByText(/2 items restored together/)).toBeTruthy();
});

it("cannot apply a late preview from a cancelled selection to a new selection", async () => {
  let finish!: (value: RecoveryBatchPreview) => void;
  vi.mocked(api.previewRecoveryBatch).mockReturnValueOnce(new Promise(resolve => { finish = resolve; })).mockResolvedValue({ ...preview, batch_id: "travel-only", items: [preview.items[1]] });
  show(); await selectBoth();
  fireEvent.click(screen.getByRole("button", { name: "Review restore selection" }));
  await waitFor(() => expect(api.previewRecoveryBatch).toHaveBeenCalledTimes(1));
  const oldSignal = vi.mocked(api.previewRecoveryBatch).mock.calls[0][1]!;
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Garden notes" }));
  const dialog = await review();
  await act(async () => { finish(preview); });
  expect(oldSignal.aborted).toBe(true);
  expect(within(dialog).queryByText("Garden notes")).toBeNull();
  expect(within(dialog).getByText("Travel notes")).toBeTruthy();
  expect(api.applyRecoveryBatch).not.toHaveBeenCalled();
});

it("caps selection at twenty and refuses nonrecoverable items without dropping chosen items", async () => {
  const items = Array.from({ length: 21 }, (_, index) => ({ ...recoveryItem(`item-${index}`), display_label: `Garden ${index}` }));
  vi.mocked(api.recoveryItems).mockResolvedValue({ items: [...items, { ...travel, state: "blocked" }], next_cursor: null });
  show(); await screen.findByText("Garden 0");
  for (let index = 0; index < 20; index++) fireEvent.click(screen.getByRole("checkbox", { name: `Select Garden ${index}` }));
  const last = screen.getByRole("checkbox", { name: "Select Garden 20" });
  expect(last).toHaveAttribute("aria-disabled", "true"); fireEvent.click(last);
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Travel notes" }));
  expect(screen.getByText("20 of 20 items selected")).toBeTruthy();
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Garden 0" }));
  expect(last).toHaveAttribute("aria-disabled", "false"); fireEvent.click(last);
  await review(); expect(api.previewRecoveryBatch).toHaveBeenCalledWith(expect.objectContaining({ deletion_ids: items.slice(1).map(item => item.deletion_id) }), expect.any(AbortSignal));
});

it("preserves selected pages and clears hidden selections when a filter changes", async () => {
  vi.mocked(api.recoveryItems).mockResolvedValueOnce({ items: [garden], next_cursor: "more" }).mockResolvedValue({ items: [travel], next_cursor: null });
  show(); fireEvent.click(await screen.findByRole("checkbox", { name: "Select Garden notes" }));
  fireEvent.click(screen.getByRole("button", { name: "Load more deleted items" }));
  fireEvent.click(await screen.findByRole("checkbox", { name: "Select Travel notes" }));
  expect(screen.getByText("2 of 20 items selected")).toBeTruthy();
  fireEvent.change(screen.getByLabelText("Type"), { target: { value: "project" } });
  expect(screen.queryByText("2 of 20 items selected")).toBeNull();
  expect(api.previewRecoveryBatch).not.toHaveBeenCalled();
});

it("checks again only on explicit request after a materialization failure", async () => {
  vi.mocked(api.previewRecoveryBatch).mockRejectedValueOnce(new Error("Database busy")).mockResolvedValue(preview);
  show(); await selectBoth(); const dialog = await review();
  expect(within(dialog).getByText("Database busy")).toBeTruthy();
  fireEvent.click(within(dialog).getByRole("button", { name: "Restore selected items" }));
  expect(api.applyRecoveryBatch).not.toHaveBeenCalled();
  fireEvent.click(within(dialog).getByRole("button", { name: "Recheck selected recovery details" }));
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Restore selected items" })).toHaveAttribute("aria-disabled", "false"));
  expect(api.previewRecoveryBatch).toHaveBeenCalledTimes(2); expect(api.applyRecoveryBatch).not.toHaveBeenCalled();
});

it("expires an unsubmitted preview at the exact deadline without extending the recovery window", async () => {
  vi.useFakeTimers(); vi.setSystemTime(new Date("2026-10-02T00:00:00Z"));
  const deadline = "2026-10-02T00:01:00Z";
  vi.mocked(api.previewRecoveryBatch).mockResolvedValue({ ...preview, expires_at: deadline });
  show(); await act(async () => { await vi.advanceTimersByTimeAsync(20); });
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Garden notes" }));
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Travel notes" }));
  fireEvent.click(screen.getByRole("button", { name: "Review restore selection" }));
  await act(async () => { await vi.advanceTimersByTimeAsync(20); });
  const confirm = screen.getByRole("button", { name: "Restore selected items" });
  expect(confirm).toHaveAttribute("aria-disabled", "false");
  await act(async () => { await vi.advanceTimersByTimeAsync(59_959); });
  expect(confirm).toHaveAttribute("aria-disabled", "false");
  await act(async () => { await vi.advanceTimersByTimeAsync(1); });
  expect(confirm).toHaveAttribute("aria-disabled", "true"); fireEvent.click(confirm);
  expect(api.applyRecoveryBatch).not.toHaveBeenCalled(); expect(api.previewRecoveryBatch).toHaveBeenCalledTimes(1);
  expect(screen.getByText(/selection preview expired/)).toBeTruthy();
});

it("refuses a preview that already expired while its response was delayed", async () => {
  vi.useFakeTimers(); vi.setSystemTime(new Date("2026-10-02T00:00:00Z"));
  let finish!: (value: RecoveryBatchPreview) => void;
  vi.mocked(api.previewRecoveryBatch).mockReturnValue(new Promise(resolve => { finish = resolve; }));
  show(); await act(async () => { await vi.advanceTimersByTimeAsync(20); });
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Garden notes" }));
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Travel notes" }));
  fireEvent.click(screen.getByRole("button", { name: "Review restore selection" }));
  await act(async () => { await vi.advanceTimersByTimeAsync(60_000); });
  await act(async () => { finish({ ...preview, expires_at: "2026-10-02T00:00:30Z" }); await vi.advanceTimersByTimeAsync(20); });
  const confirm = screen.getByRole("button", { name: "Restore selected items" });
  expect(confirm).toHaveAttribute("aria-disabled", "true"); fireEvent.click(confirm);
  expect(api.applyRecoveryBatch).not.toHaveBeenCalled();
});
