/** What an encrypted backup holds, once it has opened and passed a backup's checks. */
export interface EncryptedBackupCheck {
  created_at: string;
  app_version: string;
  schema_revision: string;
  database_size_bytes: number;
  media_included: boolean;
  media_size_bytes: number | null;
  artifact_count: number;
}
