/** Making an encrypted backup from Data & backups, and checking one later. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { EncryptedBackups } from "./EncryptedBackups";

vi.mock("./api", () => ({ api: { createEncryptedBackup: vi.fn(), checkEncryptedBackup: vi.fn() } }));

const PASSPHRASE = "correct horse battery staple";
const downloads: string[] = [];

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
  fireEvent.change(screen.getByLabelText("Encrypted backup to check"), { target: { files: [file] } });
}

beforeEach(() => {
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

  expect(screen.queryByLabelText("Passphrase of the backup to check")).toBeNull();
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
  const field = await screen.findByLabelText("Passphrase of the backup to check");
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
  fireEvent.change(await screen.findByLabelText("Passphrase of the backup to check"), {
    target: { value: "not the passphrase" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Check" }));

  expect(await screen.findByRole("alert")).toHaveTextContent(sentence);
  expect(screen.queryByText("The server's own words.")).toBeNull();
});
