/** Saying that a restore asked for could not be applied, until it is dismissed. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { RestoreFailureNotice } from "./RestoreFailureNotice";
import type { BackupRestoreState } from "./types";

vi.mock("./api", () => ({ api: { backupRestoreState: vi.fn(), dismissFailedRestore: vi.fn() } }));

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><RestoreFailureNotice /></QueryClientProvider>);
}

function failed(reason: BackupRestoreState["reason"]): BackupRestoreState {
  return { state: "failed", backup: "local-lm-20260110T120000Z-00000001.sqlite3", reason, failed_at: "2026-10-04T12:00:00Z", encrypted: false };
}

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

it.each([
  ["backup-missing", "The backup it was asked to use was no longer there."],
  ["backup-invalid", "The backup did not pass its checks."],
  ["backup-newer", "The backup was made by a newer version of LM Atelier."],
  ["backup-key-missing", "The key that opens the encrypted backup was no longer in this computer's credential vault."],
  ["restore-failed", "The backup could not be put in place."],
] as const)("says the data was left as it was, and why: %s", async (reason, sentence) => {
  vi.mocked(api.backupRestoreState).mockResolvedValue(failed(reason));
  show();

  const notice = await screen.findByRole("status");
  expect(notice).toHaveTextContent("your data was left as it was");
  expect(notice).toHaveTextContent(sentence);
});

it.each([
  { state: "none", backup: null, reason: null, failed_at: null, encrypted: false },
  { state: "pending", backup: "local-lm-20260110T120000Z-00000001.sqlite3", reason: null, failed_at: null, encrypted: false },
  { state: "pending", backup: null, reason: null, failed_at: null, encrypted: true },
] as const)("says nothing when no restore failed: $state", async (state) => {
  vi.mocked(api.backupRestoreState).mockResolvedValue(state);
  show();

  await waitFor(() => expect(api.backupRestoreState).toHaveBeenCalled());
  expect(screen.queryByRole("status")).toBeNull();
});

it("goes away once dismissed", async () => {
  vi.mocked(api.backupRestoreState)
    .mockResolvedValueOnce(failed("backup-missing"))
    .mockResolvedValue({ state: "none", backup: null, reason: null, failed_at: null, encrypted: false });
  vi.mocked(api.dismissFailedRestore).mockResolvedValue(undefined);
  show();

  fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }));

  await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
  expect(api.dismissFailedRestore).toHaveBeenCalledTimes(1);
});

it("says so when it cannot be dismissed, and stays", async () => {
  vi.mocked(api.backupRestoreState).mockResolvedValue(failed("backup-missing"));
  vi.mocked(api.dismissFailedRestore).mockRejectedValue(new Error("neutral internal error marker"));
  show();

  fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("The notice could not be dismissed. Try again.");
  expect(screen.queryByText(/neutral internal error marker/)).toBeNull();
  expect(screen.getByRole("status")).toHaveTextContent("your data was left as it was");
});
