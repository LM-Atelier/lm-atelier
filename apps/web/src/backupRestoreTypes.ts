/** A restore waiting for the next start, or why the last one asked for was not applied. */
export interface BackupRestoreState {
  state: "none" | "pending" | "failed";
  backup: string | null;
  reason: "backup-missing" | "backup-invalid" | "backup-newer" | "backup-key-missing" | "restore-failed" | null;
  failed_at: string | null;
  /** Whether the restore is, or was, of an encrypted backup file rather than a backup here. */
  encrypted: boolean;
}
