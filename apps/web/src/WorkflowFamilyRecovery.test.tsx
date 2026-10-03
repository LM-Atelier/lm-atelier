import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { api, ApiError } from "./api";
import { WorkflowFamilyRecovery } from "./WorkflowFamilyRecovery";
import { recoveryCommand, recoveryImpact, recoveryItem } from "./test/recoveryFixtures";
import type { WorkflowFamily } from "./types";
import type { RecoveryImpact, RecoveryItem, RecoveryResult } from "./recoveryTypes";

vi.mock("./api", async (original) => ({ ...(await original<typeof import("./api")>()), api: {
  workflowDeletionImpact: vi.fn(), trashWorkflow: vi.fn(), recoveryImpact: vi.fn(), restoreRecovery: vi.fn(),
} }));
const family: WorkflowFamily = { id: "family-garden", name: "Garden", description: "", use_case: "", tags: [],
  enabled: true, archived: false, compatibility: false, created_at: "2026-09-08T00:00:00Z", updated_at: "2026-09-08T00:00:00Z",
  preferences: [], variants: [] };
const item: RecoveryItem = { ...recoveryItem(family.id), kind: "workflow_family", display_label: family.name };
const result: RecoveryResult = { kind: "workflow_family", subject_id: family.id,
  deletion_id: item.deletion_id, action: "restore", replayed: false, reclaimed_bytes: 0 };
const preview: RecoveryImpact = { ...recoveryImpact(family.id), kind: "workflow_family",
  counts: { ...item.counts, workflow_definitions: 2, workflow_revisions: 4 } };
let client: QueryClient;
const selection = vi.fn();
function mount(selectedId: string | null = "garden", value = family) {
  const element = (id: string | null) => <QueryClientProvider client={client}>
    <WorkflowFamilyRecovery family={value} selectedId={id} onSelectionChange={selection} />
  </QueryClientProvider>;
  const view = render(element(selectedId));
  return { ...view, select: (id: string | null) => view.rerender(element(id)) };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
async function open() {
  fireEvent.click(screen.getByRole("button", { name: "Delete workflow family" }));
  return screen.findByRole("dialog", { name: "Move this workflow family to Recently Deleted?" });
}
async function trash() {
  const dialog = await open();
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Move to Recently Deleted" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  await screen.findByRole("button", { name: "Undo" });
}
beforeEach(() => {
  vi.resetAllMocks();
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  vi.mocked(api.workflowDeletionImpact).mockResolvedValue(preview);
  vi.mocked(api.trashWorkflow).mockResolvedValue(item);
  vi.mocked(api.recoveryImpact).mockResolvedValue({ ...preview, available_actions: ["restore"] });
  vi.mocked(api.restoreRecovery).mockResolvedValue(result);
});
afterEach(() => { cleanup(); client.clear(); vi.useRealTimers(); });

it("waits for a fresh preview and cancels without deleting or changing archive state", async () => {
  const pending = deferred<RecoveryImpact>();
  vi.mocked(api.workflowDeletionImpact).mockReturnValue(pending.promise);
  mount(); const dialog = await open();
  const confirm = within(dialog).getByRole("button", { name: "Move to Recently Deleted" });
  expect(confirm).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(confirm); expect(api.trashWorkflow).not.toHaveBeenCalled();
  await act(async () => pending.resolve(preview));
  expect(await within(dialog).findByText(/2 variants and 4 revisions/)).toBeInTheDocument();
  expect(within(dialog).queryByText("Cannot be undone")).not.toBeInTheDocument();
  fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(api.trashWorkflow).not.toHaveBeenCalled(); expect(selection).not.toHaveBeenCalled();
});

it.each([
  ["active_selection", "This family is selected by a chat or project, or is set as a default."],
  ["active_work", "Queued, held or running work still needs this family."],
] as const)("explains the %s blocker and never submits the blocked deletion", async (conflict, explanation) => {
  vi.mocked(api.workflowDeletionImpact).mockResolvedValue({ ...preview, conflicts: [conflict], available_actions: [] });
  mount(); const dialog = await open();
  expect(await within(dialog).findByRole("alert")).toHaveTextContent(explanation);
  const confirm = within(dialog).getByRole("button", { name: "Move to Recently Deleted" });
  expect(confirm).toHaveAttribute("aria-disabled", "true"); fireEvent.click(confirm);
  expect(api.trashWorkflow).not.toHaveBeenCalled(); expect(selection).not.toHaveBeenCalled();
});

it("retries an uncertain Trash response with exactly the same frozen command", async () => {
  vi.mocked(api.trashWorkflow).mockRejectedValueOnce(new Error("Connection interrupted"));
  mount(); const dialog = await open();
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Move to Recently Deleted" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  fireEvent.click(await screen.findByRole("button", { name: "Try again" }));
  await screen.findByRole("button", { name: "Undo" });
  expect(api.workflowDeletionImpact).toHaveBeenCalledOnce();
  expect(api.trashWorkflow).toHaveBeenCalledTimes(2);
  expect(vi.mocked(api.trashWorkflow).mock.calls[1]).toEqual(vi.mocked(api.trashWorkflow).mock.calls[0]);
  expect(vi.mocked(api.trashWorkflow).mock.calls[0][1]).toMatchObject({
    expected_revision: recoveryCommand.expected_revision, impact_sha256: recoveryCommand.impact_sha256,
  });
});

it("requires an explicit new preview and operation identity after a stale Trash refusal", async () => {
  vi.mocked(api.trashWorkflow).mockRejectedValueOnce(new ApiError(409, undefined, "Workflow changed", "recovery-stale"));
  mount(); const dialog = await open();
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Move to Recently Deleted" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  await within(dialog).findByText("Workflow changed");
  const confirm = within(dialog).getByRole("button", { name: "Move to Recently Deleted" });
  expect(confirm).toHaveAttribute("aria-disabled", "true"); fireEvent.click(confirm);
  expect(api.trashWorkflow).toHaveBeenCalledOnce();
  vi.mocked(api.workflowDeletionImpact).mockResolvedValue({ ...preview, revision: "d".repeat(64), impact_sha256: "e".repeat(64) });
  fireEvent.click(within(dialog).getByRole("button", { name: "Check again" }));
  await waitFor(() => expect(confirm).toHaveAttribute("aria-disabled", "false")); fireEvent.click(confirm);
  await screen.findByRole("button", { name: "Undo" });
  expect(api.workflowDeletionImpact).toHaveBeenCalledTimes(2);
  const calls = vi.mocked(api.trashWorkflow).mock.calls;
  expect(calls[1][1]).toMatchObject({ expected_revision: "d".repeat(64), impact_sha256: "e".repeat(64) });
  expect(calls[1][1].operation_key).not.toBe(calls[0][1].operation_key);
});

it("keeps a pending Trash dialog open and submits only once", async () => {
  const pending = deferred<RecoveryItem>(); vi.mocked(api.trashWorkflow).mockReturnValue(pending.promise);
  mount(); const dialog = await open();
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Move to Recently Deleted" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  fireEvent.click(within(dialog).getByRole("button", { name: "Moving workflow…" }));
  fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
  fireEvent.keyDown(dialog, { key: "Escape" });
  expect(dialog).toBeInTheDocument(); await waitFor(() => expect(api.trashWorkflow).toHaveBeenCalledOnce());
  await act(async () => pending.resolve(item)); await screen.findByRole("button", { name: "Undo" });
});

it("does not clear a different workflow opened while Trash is pending", async () => {
  const pending = deferred<RecoveryItem>(); vi.mocked(api.trashWorkflow).mockReturnValue(pending.promise);
  const view = mount(); const dialog = await open();
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Move to Recently Deleted" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  view.select("other"); await act(async () => pending.resolve(item));
  await screen.findByRole("button", { name: "Undo" }); expect(selection).not.toHaveBeenCalled();
});

it("clears another selected variant of the same deleted family", async () => {
  const pending = deferred<RecoveryItem>(); vi.mocked(api.trashWorkflow).mockReturnValue(pending.promise);
  const sibling: WorkflowFamily["variants"][number] = { id: "sibling", name: "Garden video", variant_key: "video", operation: "text_to_video",
    current_revision_id: null, current_revision_version: null, engine: null, capabilities: [], trusted: false,
    readiness: "unavailable", readiness_reason: null };
  const view = mount("garden", { ...family, variants: [sibling] }); const dialog = await open();
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Move to Recently Deleted" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(within(dialog).getByRole("button", { name: "Move to Recently Deleted" }));
  view.select("sibling"); await act(async () => pending.resolve(item));
  await screen.findByRole("button", { name: "Undo" }); expect(selection).toHaveBeenCalledWith(null);
});

it("retries uncertain Undo with one preview and the same command without dismissing while pending", async () => {
  const pending = deferred<RecoveryResult>();
  vi.mocked(api.restoreRecovery).mockRejectedValueOnce(new Error("Connection interrupted")).mockReturnValueOnce(pending.promise);
  const view = mount(); await trash(); view.select(null); selection.mockClear();
  fireEvent.click(screen.getByRole("button", { name: "Undo" }));
  fireEvent.click(await screen.findByRole("button", { name: "Try Undo again" }));
  await waitFor(() => expect(api.restoreRecovery).toHaveBeenCalledTimes(2));
  fireEvent.click(screen.getByRole("button", { name: "Restoring…" }));
  fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
  expect(screen.getByText(/Undo restores this family disabled/)).toBeInTheDocument();
  expect(api.recoveryImpact).toHaveBeenCalledOnce();
  expect(vi.mocked(api.restoreRecovery).mock.calls[1]).toEqual(vi.mocked(api.restoreRecovery).mock.calls[0]);
  await act(async () => pending.resolve(result));
  await waitFor(() => expect(selection).toHaveBeenCalledWith("garden"));
  expect(screen.queryByText(/Undo restores this family disabled/)).not.toBeInTheDocument();
});

it("requires explicit Undo recheck after a stale response and keeps a newly opened selection", async () => {
  const pending = deferred<RecoveryResult>();
  vi.mocked(api.restoreRecovery).mockRejectedValueOnce(new ApiError(409, undefined, "Workflow changed", "recovery-stale")).mockReturnValueOnce(pending.promise);
  const view = mount(); await trash(); view.select(null); selection.mockClear();
  fireEvent.click(screen.getByRole("button", { name: "Undo" }));
  const recheck = await screen.findByRole("button", { name: "Recheck Undo" });
  fireEvent.click(screen.getByRole("button", { name: "Try Undo again" })); expect(api.restoreRecovery).toHaveBeenCalledOnce();
  vi.mocked(api.recoveryImpact).mockResolvedValue({ ...preview, available_actions: ["restore"], revision: "d".repeat(64) });
  fireEvent.click(recheck); fireEvent.click(await screen.findByRole("button", { name: "Undo" }));
  await waitFor(() => expect(api.restoreRecovery).toHaveBeenCalledTimes(2));
  view.select("other"); await act(async () => pending.resolve(result));
  await waitFor(() => expect(screen.queryByText(/Undo restores this family disabled/)).not.toBeInTheDocument());
  expect(selection).not.toHaveBeenCalled(); expect(api.recoveryImpact).toHaveBeenCalledTimes(2);
  const calls = vi.mocked(api.restoreRecovery).mock.calls;
  expect(calls[1][1].expected_revision).toBe("d".repeat(64));
  expect(calls[1][1].operation_key).not.toBe(calls[0][1].operation_key);
});

it("shows the original expiry and disables Undo at its exact deadline", async () => {
  const deadline = Date.now() + 1_500;
  vi.mocked(api.trashWorkflow).mockResolvedValue({ ...item, purge_after: new Date(deadline).toISOString() });
  mount(); await trash();
  expect(screen.getByText(/Recover until/).querySelector("time")).toHaveAttribute("datetime", new Date(deadline).toISOString());
  expect(screen.getByRole("button", { name: "Undo" })).toHaveAttribute("aria-disabled", "false");
  await waitFor(() => expect(screen.getByRole("button", { name: "Undo" })).toHaveAttribute("aria-disabled", "true"), { timeout: 2_500 });
  expect(Date.now()).toBeGreaterThanOrEqual(deadline);
  fireEvent.click(screen.getByRole("button", { name: "Undo" }));
  expect(api.recoveryImpact).not.toHaveBeenCalled(); expect(api.restoreRecovery).not.toHaveBeenCalled();
});

it("offers recoverable deletion for archived families without unarchiving them", async () => {
  mount("garden", { ...family, enabled: false, archived: true }); await trash();
  expect(api.trashWorkflow).toHaveBeenCalledOnce();
  expect(await screen.findByText(/Undo restores this family disabled/)).toBeInTheDocument();
});
