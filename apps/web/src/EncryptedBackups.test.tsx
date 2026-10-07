/** Making an encrypted backup from Data & backups, and checking or restoring one later. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { EncryptedBackups } from "./EncryptedBackups";
import type { BackupRestoreState, EncryptedBackupCheck } from "./types";

vi.mock("./api", () => ({
  api: {
    createEncryptedBackup: vi.fn(),
    checkEncryptedBackup: vi.fn(),
    restoreEncryptedBackup: vi.fn(),
    backupRestoreState: vi.fn(),
    cancelRestore: vi.fn(),
  },
}));

const PASSPHRASE = "correct horse battery staple";
const downloads: string[] = [];
const NOTHING_WAITING: BackupRestoreState = { state: "none", backup: null, reason: null, failed_at: null, encrypted: false };
const ENCRYPTED_WAITING: BackupRestoreState = { ...NOTHING_WAITING, state: "pending", encrypted: true };
const REPORT: EncryptedBackupCheck = {
  created_at: "2026-10-04T12:00:00Z",
  app_version: "1.2.3",
  schema_revision: "0123456789ab",
  database_size_bytes: 2048,
  media_included: true,
  media_size_bytes: 4096,
  artifact_count: 3,
};

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><EncryptedBackups /></QueryClientProvider>);
}

function refusal(code: string, message = "The server's own words.") {
  return Object.assign(new Error(message), { code });
}

// The first bytes of an encrypted archive: magic, version, suite, then its kind.
function encryptedFile(kind: number, name = "workspace.lm-atelier.encrypted") {
  return new File([new Uint8Array([0x4c, 0x4d, 0x41, 0x41, 0x52, 0x43, 0x48, 0x00, 1, 1, kind, 1, 0, 0])], name);
}

function typePassphrases(first: string, second: string) {
  fireEvent.change(screen.getByLabelText("Encrypted backup passphrase"), { target: { value: first } });
  fireEvent.change(screen.getByLabelText("Confirm encrypted backup passphrase"), { target: { value: second } });
}

function choose(file: File) {
  fireEvent.change(screen.getByLabelText("Encrypted backup to check or restore"), { target: { files: [file] } });
}

async function chooseWithPassphrase(file: File, passphrase = PASSPHRASE) {
  choose(file);
  fireEvent.change(await screen.findByLabelText("Passphrase of the backup"), { target: { value: passphrase } });
}

beforeEach(() => {
  vi.mocked(api.backupRestoreState).mockResolvedValue(NOTHING_WAITING);
  downloads.length = 0;
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    downloads.push(this.getAttribute("href") ?? "");
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("makes nothing until the passphrase is typed the same way twice", () => {
  show();
  const withMedia = screen.getByRole("button", { name: "Encrypted backup with media" });

  fireEvent.click(withMedia);
  typePassphrases(PASSPHRASE, "correct horse battery stapel");
  fireEvent.click(withMedia);

  expect(withMedia).toHaveAttribute("aria-disabled", "true");
  expect(screen.getByText("The passphrases do not match.")).toBeTruthy();
  expect(api.createEncryptedBackup).not.toHaveBeenCalled();
});

it("sends the passphrase to be sealed with, downloads the checked copy, and forgets the passphrase", async () => {
  vi.mocked(api.createEncryptedBackup).mockResolvedValue({ url: "/api/artifacts/backup-1/content" });
  show();

  typePassphrases(PASSPHRASE, PASSPHRASE);
  fireEvent.click(screen.getByRole("button", { name: "Encrypted backup with media" }));

  await waitFor(() => expect(downloads).toEqual(["/api/artifacts/backup-1/content"]));
  expect(api.createEncryptedBackup).toHaveBeenCalledWith(true, PASSPHRASE);
  expect(downloads[0]).not.toContain(encodeURIComponent(PASSPHRASE));
  expect(screen.getByRole("status")).toHaveTextContent("opened again to check it");
  expect(screen.getByLabelText("Encrypted backup passphrase")).toHaveValue("");
  expect(screen.getByLabelText("Confirm encrypted backup passphrase")).toHaveValue("");
});

it.each([
  ["archive-passphrase-invalid", "A passphrase can be at most 1024 bytes. Choose a shorter one."],
  ["backup-storage-insufficient", "There is not enough free disk space for this backup. Nothing was written."],
  ["backup-too-large", "This workspace is larger than an encrypted backup can hold."],
  ["encrypted-backup-busy", "An encrypted backup is already being made or checked. Try again when it finishes."],
  [
    "backup-export-unverified",
    "The encrypted backup did not open again after it was written, so it was not kept. Try again.",
  ],
  ["something-new", "The encrypted backup could not be made. Try again."],
])("says why a backup was not made in its own words: %s", async (code, sentence) => {
  vi.mocked(api.createEncryptedBackup).mockRejectedValue(refusal(code));
  show();

  typePassphrases(PASSPHRASE, PASSPHRASE);
  fireEvent.click(screen.getByRole("button", { name: "Encrypted backup of state" }));

  expect(await screen.findByRole("alert")).toHaveTextContent(sentence);
  expect(screen.queryByText("The server's own words.")).toBeNull();
  expect(api.createEncryptedBackup).toHaveBeenCalledWith(false, PASSPHRASE);
  expect(downloads).toEqual([]);
});

it("never sends a file that is not an encrypted backup", async () => {
  show();

  choose(encryptedFile(1, "garden.lm-atelier.encrypted"));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "garden.lm-atelier.encrypted is not an encrypted LM Atelier backup.",
  );
  choose(new File(["plain text"], "notes.txt"));
  // The earlier alert stays until this one replaces it, so wait for the new words.
  expect(await screen.findByText("notes.txt is not an encrypted LM Atelier backup.")).toBeTruthy();

  expect(screen.queryByLabelText("Passphrase of the backup")).toBeNull();
  expect(api.checkEncryptedBackup).not.toHaveBeenCalled();
});

it("checks a chosen backup with its passphrase and reports what it holds", async () => {
  vi.mocked(api.checkEncryptedBackup).mockResolvedValue({
    created_at: "2026-10-04T12:00:00Z",
    app_version: "1.2.3",
    schema_revision: "0123456789ab",
    database_size_bytes: 2048,
    media_included: true,
    media_size_bytes: 4096,
    artifact_count: 3,
  });
  show();
  const backup = encryptedFile(3);

  choose(backup);
  const field = await screen.findByLabelText("Passphrase of the backup");
  await waitFor(() => expect(field).toHaveFocus());
  fireEvent.click(screen.getByRole("button", { name: "Check" }));
  expect(api.checkEncryptedBackup).not.toHaveBeenCalled();
  fireEvent.change(field, { target: { value: PASSPHRASE } });
  fireEvent.click(screen.getByRole("button", { name: "Check" }));

  const report = await screen.findByText(/passed every check/);
  expect(report).toHaveTextContent("version 1.2.3");
  expect(report).toHaveTextContent("with 3 pictures and videos");
  expect(api.checkEncryptedBackup).toHaveBeenCalledWith(backup, PASSPHRASE);
  expect(field).toHaveValue("");
});

it.each([
  ["archive-passphrase-or-archive-invalid", "The passphrase is wrong, or the file is damaged."],
  ["backup-invalid", "The file opened, but it does not hold a complete LM Atelier backup."],
  [
    "archive-key-derivation-failed",
    "This computer could not set aside the memory the passphrase needs. Close other applications and try again.",
  ],
  ["backup-file-too-large", "This file is larger than an encrypted backup can be."],
])("says why a backup did not check in its own words: %s", async (code, sentence) => {
  vi.mocked(api.checkEncryptedBackup).mockRejectedValue(refusal(code));
  show();

  choose(encryptedFile(3));
  fireEvent.change(await screen.findByLabelText("Passphrase of the backup"), {
    target: { value: "not the passphrase" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Check" }));

  expect(await screen.findByRole("alert")).toHaveTextContent(sentence);
  expect(screen.queryByText("The server's own words.")).toBeNull();
});

it("restores a chosen backup on restart only once that is confirmed, and says what will happen", async () => {
  vi.mocked(api.restoreEncryptedBackup).mockResolvedValue(REPORT);
  show();
  const backup = encryptedFile(3);
  await chooseWithPassphrase(backup);

  fireEvent.click(screen.getByRole("button", { name: "Restore on restart" }));
  const question = await screen.findByRole("dialog", { name: "Restore this encrypted backup on restart?" });
  expect(question).toHaveTextContent("the passphrase itself is not kept");
  expect(api.restoreEncryptedBackup).not.toHaveBeenCalled();
  vi.mocked(api.backupRestoreState).mockResolvedValue(ENCRYPTED_WAITING);
  fireEvent.click(within(question).getByRole("button", { name: "Restore on restart" }));

  expect(await screen.findByText(/Restart LM Atelier to apply it/)).toBeTruthy();
  expect(screen.getByText(/passed every check/)).toHaveTextContent("with 3 pictures and videos");
  expect(api.restoreEncryptedBackup).toHaveBeenCalledWith(backup, PASSPHRASE);
  expect(screen.getByLabelText("Passphrase of the backup")).toHaveValue("");
  // The restore state is asked for again, so the waiting restore shows at once.
  expect(await screen.findByRole("button", { name: "Cancel restore" })).toBeTruthy();
});

it("sends nothing when the restore is not confirmed", async () => {
  show();
  await chooseWithPassphrase(encryptedFile(3));

  fireEvent.click(screen.getByRole("button", { name: "Restore on restart" }));
  const question = await screen.findByRole("dialog", { name: "Restore this encrypted backup on restart?" });
  fireEvent.click(within(question).getByRole("button", { name: "Cancel" }));

  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(api.restoreEncryptedBackup).not.toHaveBeenCalled();
  expect(screen.getByLabelText("Passphrase of the backup")).toHaveValue(PASSPHRASE);
});

it.each([
  [
    "restore-needs-key-vault",
    "Restoring an encrypted backup needs this computer's credential vault, which is not available. Nothing was scheduled.",
  ],
  ["backup-newer", "This backup was made by a newer version of LM Atelier, so this version cannot restore it."],
  ["archive-passphrase-or-archive-invalid", "The passphrase is wrong, or the file is damaged."],
  ["something-new", "The restore could not be scheduled. Try again."],
])("says why a restore was not scheduled in its own words: %s", async (code, sentence) => {
  vi.mocked(api.restoreEncryptedBackup).mockRejectedValue(refusal(code));
  show();
  await chooseWithPassphrase(encryptedFile(3));

  fireEvent.click(screen.getByRole("button", { name: "Restore on restart" }));
  const question = await screen.findByRole("dialog", { name: "Restore this encrypted backup on restart?" });
  fireEvent.click(within(question).getByRole("button", { name: "Restore on restart" }));

  expect(await screen.findByRole("alert")).toHaveTextContent(sentence);
  expect(screen.queryByText("The server's own words.")).toBeNull();
  expect(screen.queryByText(/Restart LM Atelier to apply it/)).toBeNull();
});

it("shows an encrypted restore that is waiting and cancels it", async () => {
  vi.mocked(api.backupRestoreState).mockResolvedValue(ENCRYPTED_WAITING);
  vi.mocked(api.cancelRestore).mockResolvedValue(undefined);
  show();

  const notice = await screen.findByText(/will replace the current data the next time LM Atelier starts/);
  vi.mocked(api.backupRestoreState).mockResolvedValue(NOTHING_WAITING);
  fireEvent.click(screen.getByRole("button", { name: "Cancel restore" }));

  await waitFor(() => expect(notice).not.toBeInTheDocument());
  expect(api.cancelRestore).toHaveBeenCalledTimes(1);
});

it("says so when a waiting restore cannot be cancelled, and keeps showing it", async () => {
  vi.mocked(api.backupRestoreState).mockResolvedValue(ENCRYPTED_WAITING);
  vi.mocked(api.cancelRestore).mockRejectedValue(new Error("The server's own words."));
  show();

  fireEvent.click(await screen.findByRole("button", { name: "Cancel restore" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("The restore could not be cancelled. Try again.");
  expect(screen.getByRole("button", { name: "Cancel restore" })).toBeTruthy();
});

it("leaves a waiting restore of a backup on this computer to the backup list", async () => {
  vi.mocked(api.backupRestoreState).mockResolvedValue({
    ...NOTHING_WAITING,
    state: "pending",
    backup: "local-lm-20260110T120000Z-00000001.sqlite3",
  });
  show();

  await waitFor(() => expect(api.backupRestoreState).toHaveBeenCalled());
  expect(screen.queryByRole("button", { name: "Cancel restore" })).toBeNull();
});
