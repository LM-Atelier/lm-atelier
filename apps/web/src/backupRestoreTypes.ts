/** A restore waiting for the next start, or why the last one asked for was not applied. */
export interface BackupRestoreState {
  state: "none" | "pending" | "failed";
  backup: string | null;
  reason: "backup-missing" | "backup-invalid" | "backup-newer" | "restore-failed" | null;
  failed_at: string | null;
}
