import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";
import { ChatTrashConfirmation } from "./ChatTrashConfirmation";
import { recoveryImpact } from "./test/recoveryFixtures";
import type { RecoveryImpact } from "./recoveryTypes";

vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ApiError: actual.ApiError, api: { deletionImpact: vi.fn(), cancelAllChatWork: vi.fn() } };
});
const clients: QueryClient[] = [];
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.deletionImpact).mockResolvedValue(recoveryImpact());
  vi.mocked(api.cancelAllChatWork).mockResolvedValue(undefined);
});
afterEach(() => { cleanup(); vi.useRealTimers(); for (const client of clients.splice(0)) client.clear(); });

function show(deleteGeneratedMedia = false) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  const confirm = vi.fn(), cancel = vi.fn();
  render(<QueryClientProvider client={client}><ChatTrashConfirmation chatId="chat-1" title="Garden notes"
    deleteGeneratedMedia={deleteGeneratedMedia} onConfirm={confirm} onCancel={cancel} /></QueryClientProvider>);
  return { confirm, cancel };
}

function withWork(revision = "b".repeat(64)): RecoveryImpact {
  return { ...recoveryImpact(), revision, available_actions: [], conflicts: ["active_work"] };
}
const afterWork = (): RecoveryImpact => ({ ...recoveryImpact(), revision: "d".repeat(64) });
const settle = (milliseconds = 0) => act(async () => { await vi.advanceTimersByTimeAsync(milliseconds); });
async function waitOutTheWarning() {
  await settle();
  for (let second = 0; second < 3; second += 1) await settle(1_000);
}
const cancelAndMove = () => screen.getByRole("button", { name: "Cancel its work and move to Recently Deleted" });

it("moves the chat on a confirmation given before its deletion details arrive", async () => {
  let finish!: (value: RecoveryImpact) => void;
  vi.mocked(api.deletionImpact).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  const { confirm } = show();
  const button = screen.getByRole("button", { name: "Move to Recently Deleted" });
  expect(button).toHaveAttribute("aria-disabled", "false");
  fireEvent.click(button);
  await screen.findByRole("button", { name: "Moving chat…" });
  expect(confirm).not.toHaveBeenCalled();
  finish(recoveryImpact());
  await waitFor(() => expect(confirm).toHaveBeenCalledWith({ expected_revision: "b".repeat(64), impact_sha256: "c".repeat(64), operation_key: expect.any(String) }));
  expect(api.deletionImpact).toHaveBeenCalledTimes(1);
});

it("withdraws an early confirmation when the dialog is cancelled before the details arrive", async () => {
  let finish!: (value: RecoveryImpact) => void;
  vi.mocked(api.deletionImpact).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  const { confirm, cancel } = show();
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(cancel).toHaveBeenCalledTimes(1);
  finish(recoveryImpact());
  await waitFor(() => expect(screen.getByRole("button", { name: "Move to Recently Deleted" })).toHaveAttribute("aria-disabled", "false"));
  expect(confirm).not.toHaveBeenCalled();
});

it("never cancels work on a confirmation given before the details showed any", async () => {
  vi.useFakeTimers();
  let finish!: (value: RecoveryImpact) => void;
  vi.mocked(api.deletionImpact).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  const { confirm } = show();
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  finish(withWork());
  await settle();
  expect(screen.getByRole("alert")).toHaveTextContent("This chat has work queued or running.");
  expect(cancelAndMove()).toHaveAttribute("aria-disabled", "true");
  // Waiting out the warning only unlocks the button; the work goes only when it is clicked.
  for (let second = 0; second < 4; second += 1) await settle(1_000);
  expect(cancelAndMove()).toHaveAttribute("aria-disabled", "false");
  expect(api.cancelAllChatWork).not.toHaveBeenCalled();
  expect(confirm).not.toHaveBeenCalled();
});

it("ignores a Cancel that arrives after an early confirmation has already sent the move", async () => {
  let finish!: (value: RecoveryImpact) => void;
  vi.mocked(api.deletionImpact).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  const { confirm, cancel } = show();
  confirm.mockReturnValue(new Promise<void>(() => {}));
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  const button = await screen.findByRole("button", { name: "Moving chat…" });
  expect(button).toBeTruthy();
  finish(recoveryImpact());
  // The details arrive and the move is sent before React renders again.
  for (let tick = 0; tick < 500 && confirm.mock.calls.length === 0; tick += 1) await Promise.resolve();
  expect(confirm).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(cancel).not.toHaveBeenCalled();
});

it("warns that a chat's work will be cancelled, waits three seconds, then cancels it all and moves the chat", async () => {
  vi.useFakeTimers();
  vi.mocked(api.deletionImpact).mockResolvedValueOnce(withWork()).mockResolvedValue(afterWork());
  const { confirm } = show();
  await settle();
  expect(screen.getByRole("alert")).toHaveTextContent("Moving it to Recently Deleted cancels all of that work first");
  const button = cancelAndMove();
  expect(button).toHaveAttribute("aria-disabled", "true");
  expect(screen.getByText("You can confirm in 3 seconds.")).toBeTruthy();
  fireEvent.click(button);
  await settle(1_000);
  expect(screen.getByText("You can confirm in 2 seconds.")).toBeTruthy();
  await settle(1_000);
  expect(screen.getByText("You can confirm in 1 second.")).toBeTruthy();
  expect(button).toHaveAttribute("aria-disabled", "true");
  await settle(1_000);
  expect(screen.queryByText(/You can confirm in/)).toBeNull();
  expect(button).toHaveAttribute("aria-disabled", "false");
  expect(api.cancelAllChatWork).not.toHaveBeenCalled();

  fireEvent.click(button);
  await settle();

  expect(api.cancelAllChatWork).toHaveBeenCalledWith("chat-1");
  expect(api.deletionImpact).toHaveBeenCalledTimes(2);
  // The chat is moved as it stands once its work ended, never as it was before.
  expect(vi.mocked(api.cancelAllChatWork).mock.invocationCallOrder[0])
    .toBeLessThan(vi.mocked(api.deletionImpact).mock.invocationCallOrder[1]);
  expect(confirm).toHaveBeenCalledWith(expect.objectContaining({ expected_revision: "d".repeat(64) }));
});

it("holds the dialog open while the chat's work is cancelled", async () => {
  vi.useFakeTimers();
  vi.mocked(api.deletionImpact).mockResolvedValueOnce(withWork()).mockResolvedValue(afterWork());
  let finish!: () => void;
  vi.mocked(api.cancelAllChatWork).mockReturnValue(new Promise<void>((resolve) => { finish = resolve; }));
  const { confirm, cancel } = show();
  await waitOutTheWarning();
  fireEvent.click(cancelAndMove());
  await settle();
  fireEvent.click(screen.getByRole("button", { name: "Cancelling its work…" }));
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(cancel).not.toHaveBeenCalled();
  expect(api.cancelAllChatWork).toHaveBeenCalledTimes(1);
  finish();
  await settle();
  expect(confirm).toHaveBeenCalledTimes(1);
});

it("keeps the chat and warns again when new work starts while its work is cancelled", async () => {
  vi.useFakeTimers();
  vi.mocked(api.deletionImpact).mockResolvedValueOnce(withWork()).mockResolvedValue(withWork("e".repeat(64)));
  const { confirm } = show();
  await waitOutTheWarning();
  fireEvent.click(cancelAndMove());
  await settle();
  expect(confirm).not.toHaveBeenCalled();
  expect(screen.getByText("New work started in this chat while its work was being cancelled.")).toBeTruthy();
  expect(screen.getByText("You can confirm in 3 seconds.")).toBeTruthy();
  expect(cancelAndMove()).toHaveAttribute("aria-disabled", "true");
});

it("reports a cancellation that failed and offers it again without another wait", async () => {
  vi.useFakeTimers();
  vi.mocked(api.deletionImpact).mockResolvedValueOnce(withWork()).mockResolvedValue(afterWork());
  vi.mocked(api.cancelAllChatWork).mockRejectedValueOnce(new Error("Connection interrupted."));
  const { confirm } = show();
  await waitOutTheWarning();
  fireEvent.click(cancelAndMove());
  await settle();
  expect(screen.getByText("Connection interrupted.")).toBeTruthy();
  expect(confirm).not.toHaveBeenCalled();
  expect(cancelAndMove()).toHaveAttribute("aria-disabled", "false");
  fireEvent.click(cancelAndMove());
  await settle();
  expect(api.cancelAllChatWork).toHaveBeenCalledTimes(2);
  expect(confirm).toHaveBeenCalledWith(expect.objectContaining({ expected_revision: "d".repeat(64) }));
});

it("binds the selected media choice to its preview and explains the recovery window", async () => {
  const { confirm } = show(true);
  await waitFor(() => expect(screen.getByText(/0 messages · 0 media files/)).toBeTruthy());
  expect(api.deletionImpact).toHaveBeenCalledWith("chat-1", true, expect.anything());
  expect(screen.getByText(/You can restore it for 30 days/)).toBeTruthy();
  expect(screen.getByText(/No media bytes are removed now/)).toBeTruthy();
  expect(screen.getByText(/At permanent deletion, exclusive generated media also leaves the Media Library/)).toBeTruthy();
  expect(screen.getByText(/Favorites and media retained elsewhere stay protected/)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  await waitFor(() => expect(confirm).toHaveBeenCalledTimes(1));
});

it("reports a failed check and permits cancellation without deletion", async () => {
  vi.mocked(api.deletionImpact).mockRejectedValue(new Error("The database is busy."));
  const { confirm, cancel } = show();
  await screen.findByText("The database is busy.");
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  expect(confirm).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(cancel).toHaveBeenCalledTimes(1);
});

it("keeps a stale deletion open and requires an explicit fresh check before confirming again", async () => {
  const { confirm, cancel } = show();
  confirm.mockRejectedValueOnce(new ApiError(409, undefined, "The conversation changed.", "recovery-impact-stale"));
  const button = screen.getByRole("button", { name: "Move to Recently Deleted" });
  await screen.findByText(/0 messages · 0 media files/);
  fireEvent.click(button);
  await screen.findByText("The conversation changed.");
  expect(cancel).not.toHaveBeenCalled();
  expect(button).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(button);
  expect(confirm).toHaveBeenCalledTimes(1);
  vi.mocked(api.deletionImpact).mockResolvedValue({ ...recoveryImpact(), revision: "d".repeat(64) });
  fireEvent.click(screen.getByRole("button", { name: "Check again" }));
  await waitFor(() => expect(button).toHaveAttribute("aria-disabled", "false"));
  expect(confirm).toHaveBeenCalledTimes(1);
  fireEvent.click(button);
  await waitFor(() => expect(confirm).toHaveBeenLastCalledWith(expect.objectContaining({ expected_revision: "d".repeat(64) })));
  expect(confirm).toHaveBeenCalledTimes(2);
});

it("retries an uncertain deletion with the same confirmed command", async () => {
  const { confirm } = show();
  confirm.mockRejectedValueOnce(new Error("Connection interrupted."));
  const button = screen.getByRole("button", { name: "Move to Recently Deleted" });
  await screen.findByText(/0 messages · 0 media files/);
  fireEvent.click(button);
  await screen.findByText("Connection interrupted.");
  fireEvent.click(button);
  await waitFor(() => expect(confirm).toHaveBeenCalledTimes(2));
  expect(confirm.mock.calls[1]).toEqual(confirm.mock.calls[0]);
  expect(api.deletionImpact).toHaveBeenCalledTimes(1);
});

it("keeps a pending deletion open without allowing a duplicate command or cancellation", async () => {
  const { confirm, cancel } = show();
  let finish!: () => void;
  confirm.mockReturnValue(new Promise<void>((resolve) => { finish = resolve; }));
  await screen.findByText(/0 messages · 0 media files/);
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  const pending = await screen.findByRole("button", { name: "Moving chat…" });
  fireEvent.click(pending);
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(confirm).toHaveBeenCalledTimes(1);
  expect(cancel).not.toHaveBeenCalled();
  finish();
  await screen.findByRole("button", { name: "Move to Recently Deleted" });
});
