import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";
import { ChatTrashConfirmation } from "./ChatTrashConfirmation";
import { recoveryImpact } from "./test/recoveryFixtures";
import type { RecoveryImpact } from "./recoveryTypes";

vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ApiError: actual.ApiError, api: { deletionImpact: vi.fn() } };
});
const clients: QueryClient[] = [];
beforeEach(() => { vi.resetAllMocks(); vi.mocked(api.deletionImpact).mockResolvedValue(recoveryImpact()); });
afterEach(() => { cleanup(); for (const client of clients.splice(0)) client.clear(); });

function show(deleteGeneratedMedia = false) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  const confirm = vi.fn(), cancel = vi.fn();
  render(<QueryClientProvider client={client}><ChatTrashConfirmation chatId="chat-1" title="Garden notes"
    deleteGeneratedMedia={deleteGeneratedMedia} onConfirm={confirm} onCancel={cancel} /></QueryClientProvider>);
  return { confirm, cancel };
}

it("keeps deletion inert until the current impact has arrived", async () => {
  let finish!: (value: RecoveryImpact) => void;
  vi.mocked(api.deletionImpact).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  const { confirm } = show();
  const button = screen.getByRole("button", { name: "Move to Recently Deleted" });
  fireEvent.click(button);
  expect(confirm).not.toHaveBeenCalled();
  finish(recoveryImpact());
  await waitFor(() => expect(button).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(button);
  expect(confirm).toHaveBeenCalledWith({ expected_revision: "b".repeat(64), impact_sha256: "c".repeat(64), operation_key: expect.any(String) });
});

it("binds the selected media choice to its preview and explains the recovery window", async () => {
  const { confirm } = show(true);
  await waitFor(() => expect(screen.getByRole("button", { name: "Move to Recently Deleted" })).toHaveAttribute("aria-disabled", "false"));
  expect(api.deletionImpact).toHaveBeenCalledWith("chat-1", true, expect.anything());
  expect(screen.getByText(/You can restore it for 30 days/)).toBeTruthy();
  expect(screen.getByText(/No media bytes are removed now/)).toBeTruthy();
  expect(screen.getByText(/At permanent deletion, exclusive generated media also leaves the Media Library/)).toBeTruthy();
  expect(screen.getByText(/Favorites and media retained elsewhere stay protected/)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  expect(confirm).toHaveBeenCalledTimes(1);
});

it("requires a fresh check after active work ends instead of offering a stale confirmation", async () => {
  vi.mocked(api.deletionImpact).mockResolvedValueOnce({ ...recoveryImpact(), available_actions: [], conflicts: ["active_work"] })
    .mockResolvedValue({ ...recoveryImpact(), revision: "d".repeat(64), counts: { ...recoveryImpact().counts, messages: 3 } });
  const { confirm } = show();
  await screen.findByText("Wait for this chat's work to finish before deleting it.");
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  expect(confirm).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Check again" }));
  await screen.findByText(/3 messages · 0 media files/);
  expect(confirm).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  expect(confirm).toHaveBeenCalledWith(expect.objectContaining({ expected_revision: "d".repeat(64) }));
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
  await waitFor(() => expect(button).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(button);
  await screen.findByText("The conversation changed.");
  expect(cancel).not.toHaveBeenCalled();
  fireEvent.click(button);
  expect(confirm).toHaveBeenCalledTimes(1);
  vi.mocked(api.deletionImpact).mockResolvedValue({ ...recoveryImpact(), revision: "d".repeat(64) });
  fireEvent.click(screen.getByRole("button", { name: "Check again" }));
  await waitFor(() => expect(button).toHaveAttribute("aria-disabled", "false"));
  expect(confirm).toHaveBeenCalledTimes(1);
  fireEvent.click(button);
  expect(confirm).toHaveBeenLastCalledWith(expect.objectContaining({ expected_revision: "d".repeat(64) }));
});

it("retries an uncertain deletion with the same confirmed command", async () => {
  const { confirm } = show();
  confirm.mockRejectedValueOnce(new Error("Connection interrupted."));
  const button = screen.getByRole("button", { name: "Move to Recently Deleted" });
  await waitFor(() => expect(button).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(button);
  await screen.findByText("Connection interrupted.");
  fireEvent.click(button);
  expect(confirm.mock.calls[1]).toEqual(confirm.mock.calls[0]);
  expect(api.deletionImpact).toHaveBeenCalledTimes(1);
});

it("keeps a pending deletion open without allowing a duplicate command or cancellation", async () => {
  const { confirm, cancel } = show();
  let finish!: () => void;
  confirm.mockReturnValue(new Promise<void>((resolve) => { finish = resolve; }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Move to Recently Deleted" })).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(screen.getByRole("button", { name: "Move to Recently Deleted" }));
  const pending = await screen.findByRole("button", { name: "Moving chat…" });
  fireEvent.click(pending);
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(confirm).toHaveBeenCalledTimes(1);
  expect(cancel).not.toHaveBeenCalled();
  finish();
  await screen.findByRole("button", { name: "Move to Recently Deleted" });
});
