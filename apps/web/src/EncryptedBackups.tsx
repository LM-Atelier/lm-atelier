import { useMutation } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";
import { ENCRYPTED_BACKUP_KIND, encryptedArchiveKind } from "./projectArchiveFiles";
import type { EncryptedBackupCheck } from "./types";

// Each refusal in this page's own words, so a message never depends on the server's.
const PROBLEMS: Record<string, string> = {
  "archive-passphrase-or-archive-invalid": "The passphrase is wrong, or the file is damaged.",
  "archive-passphrase-invalid": "A passphrase can be at most 1024 bytes. Choose a shorter one.",
  "archive-passphrase-required": "Enter the backup's passphrase to check it.",
  "archive-kind-mismatch": "This file is not an encrypted LM Atelier backup.",
  "archive-format-unsupported": "This file's format is not one this version can open.",
  "archive-limits-exceeded": "This file asks for settings outside what this version allows.",
  "archive-key-derivation-failed":
    "This computer could not set aside the memory the passphrase needs. Close other applications and try again.",
  "archive-staging-not-private":
    "The data folder does not allow a folder private to your account, so nothing was written.",
  "encrypted-backup-busy": "An encrypted backup is already being made or checked. Try again when it finishes.",
  "backup-storage-insufficient": "There is not enough free disk space for this backup. Nothing was written.",
  "backup-too-large": "This workspace is larger than an encrypted backup can hold.",
  "backup-file-too-large": "This file is larger than an encrypted backup can be.",
  "backup-invalid": "The file opened, but it does not hold a complete LM Atelier backup.",
  "backup-export-unverified":
    "The encrypted backup did not open again after it was written, so it was not kept. Try again.",
};

function problem(error: Error, fallback: string): string {
  const code = (error as { code?: unknown }).code;
  return (typeof code === "string" && PROBLEMS[code]) || fallback;
}

function made(value: string): string {
  const when = new Date(value);
  return Number.isNaN(when.getTime())
    ? value
    : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(when);
}

function CheckReport({ report }: { report: EncryptedBackupCheck }) {
  return (
    <div className="callout success" role="status">
      This backup opens with its passphrase and passed every check. It was made {made(report.created_at)} by
      version {report.app_version} and holds {formatBytes(report.database_size_bytes)} of chats, projects and
      settings,{" "}
      {report.media_included
        ? `with ${report.artifact_count} ${report.artifact_count === 1 ? "picture or video" : "pictures and videos"} (${formatBytes(report.media_size_bytes)}).`
        : "without pictures or videos."}
    </div>
  );
}

/** A copy of this workspace sealed with a passphrase, and a check that such a copy still opens.
 *
 * The recovery backups above stay on this computer, in plain form. This copy is
 * the one to keep somewhere else: it is made fresh, sealed, opened again and
 * checked on the server before it is offered for download, and the passphrase,
 * typed twice so a slip of the keyboard does not lock it for good, is kept
 * nowhere but this page. Checking a copy later opens it and checks it the same
 * way, and restores and changes nothing.
 */
export function EncryptedBackups() {
  const [passphrase, setPassphrase] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const hint = useId();
  const picker = useRef<HTMLInputElement>(null);
  const unlockField = useRef<HTMLInputElement>(null);
  const [chosen, setChosen] = useState<File | null>(null);
  const [unlock, setUnlock] = useState("");
  const [notBackup, setNotBackup] = useState<string | null>(null);
  const create = useMutation({
    // Not kept once finished, so a passphrase it was given does not linger.
    gcTime: 0,
    mutationFn: ({ includeMedia, secret }: { includeMedia: boolean; secret: string }) =>
      api.createEncryptedBackup(includeMedia, secret),
    onSuccess: (artifact) => {
      setPassphrase("");
      setConfirmation("");
      const link = document.createElement("a");
      link.href = artifact.url;
      link.download = "";
      link.click();
    },
  });
  const check = useMutation({
    gcTime: 0,
    mutationFn: ({ file, secret }: { file: File; secret: string }) => api.checkEncryptedBackup(file, secret),
    onSuccess: () => setUnlock(""),
  });
  const ready = passphrase.length > 0 && passphrase === confirmation;
  useEffect(() => {
    if (chosen) unlockField.current?.focus();
  }, [chosen]);

  const make = (includeMedia: boolean) => {
    if (!ready || create.isPending) return;
    create.mutate({ includeMedia, secret: passphrase });
  };
  const choose = async (file: File) => {
    check.reset();
    setUnlock("");
    if ((await encryptedArchiveKind(file).catch(() => null)) !== ENCRYPTED_BACKUP_KIND) {
      setChosen(null);
      setNotBackup(`${file.name} is not an encrypted LM Atelier backup.`);
      return;
    }
    setNotBackup(null);
    setChosen(file);
  };
  const runCheck = () => {
    if (!chosen || !unlock || check.isPending) return;
    check.mutate({ file: chosen, secret: unlock });
  };
  const close = () => {
    setChosen(null);
    setUnlock("");
    check.reset();
  };

  return (
    <section aria-labelledby="encrypted-backups-heading">
      <div className="detail-title storage-actions">
        <div>
          <h2 id="encrypted-backups-heading">Encrypted backups</h2>
          <p>
            A copy of this workspace sealed with a passphrase, to keep on another drive or computer. It cannot be
            opened without the passphrase, and a forgotten passphrase cannot be recovered.
          </p>
        </div>
        <div className="row-actions storage-actions">
          <input
            ref={picker}
            hidden
            type="file"
            aria-label="Encrypted backup to check"
            accept=".lm-atelier.encrypted,.encrypted,application/octet-stream"
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) void choose(file);
            }}
          />
          <button
            type="button"
            className="secondary"
            aria-disabled={check.isPending}
            onClick={() => {
              if (!check.isPending) picker.current?.click();
            }}
          >
            Check an encrypted backup
          </button>
        </div>
      </div>
      <span className="row-actions">
        <label>
          Encrypted backup passphrase
          <input
            type="password"
            autoComplete="new-password"
            value={passphrase}
            readOnly={create.isPending}
            onChange={(event) => setPassphrase(event.target.value)}
          />
        </label>
        <label>
          Confirm encrypted backup passphrase
          <input
            type="password"
            autoComplete="new-password"
            value={confirmation}
            readOnly={create.isPending}
            onChange={(event) => setConfirmation(event.target.value)}
          />
        </label>
      </span>
      {!ready && (
        <p className="muted" id={hint}>
          {confirmation.length > 0 && passphrase !== confirmation
            ? "The passphrases do not match."
            : "Type the passphrase twice to make an encrypted backup."}
        </p>
      )}
      <span className="row-actions">
        <button
          type="button"
          className="secondary"
          aria-disabled={!ready || create.isPending}
          aria-describedby={ready ? undefined : hint}
          onClick={() => make(false)}
        >
          Encrypted backup of state
        </button>
        <button
          type="button"
          className="secondary"
          aria-disabled={!ready || create.isPending}
          aria-describedby={ready ? undefined : hint}
          onClick={() => make(true)}
        >
          Encrypted backup with media
        </button>
      </span>
      {create.isPending && (
        <p className="muted" role="status">
          Making and checking the encrypted backup. With media this can take several minutes.
        </p>
      )}
      {create.isSuccess && (
        <div className="callout success" role="status">
          The encrypted backup was made and opened again to check it. Your browser is downloading it.
        </div>
      )}
      {create.error && (
        <ErrorCallout message={problem(create.error, "The encrypted backup could not be made. Try again.")} />
      )}
      {notBackup && <ErrorCallout message={notBackup} />}
      {chosen && (
        <form
          className="callout"
          aria-label="Check an encrypted backup"
          onSubmit={(event) => {
            event.preventDefault();
            runCheck();
          }}
        >
          <p>Enter the passphrase for {chosen.name} to check that it opens. Nothing is restored or changed.</p>
          <label>
            Passphrase of the backup to check
            <input
              ref={unlockField}
              type="password"
              autoComplete="current-password"
              value={unlock}
              onChange={(event) => setUnlock(event.target.value)}
            />
          </label>
          {check.isPending && (
            <p className="muted" role="status">
              Opening and checking the backup. A large one can take several minutes.
            </p>
          )}
          {check.error && (
            <ErrorCallout message={problem(check.error, "The backup could not be checked. Try again.")} />
          )}
          {check.data && <CheckReport report={check.data} />}
          <span className="row-actions">
            <button type="submit" aria-disabled={!unlock || check.isPending}>
              Check
            </button>
            <button
              type="button"
              className="secondary"
              aria-disabled={check.isPending}
              onClick={() => {
                if (!check.isPending) close();
              }}
            >
              Close
            </button>
          </span>
        </form>
      )}
    </section>
  );
}
