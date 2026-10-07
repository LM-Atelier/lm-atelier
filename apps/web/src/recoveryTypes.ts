export type RecoveryKind = "chat" | "project" | "media_library_entry" | "workflow_family";
export type RecoveryState = "recoverable" | "restoring" | "purging" | "blocked" | "purged";
export type RecoveryAction = "trash" | "restore" | "purge";
export type RecoveryConflict = "original_project_missing" | "subject_missing" | "active_work" | "active_selection" | "record_changed";

export interface RecoveryCounts {
  chats: number;
  messages: number;
  message_parts: number;
  response_revisions: number;
  runs: number;
  jobs: number;
  work_plans: number;
  workflow_families: number;
  workflow_definitions: number;
  workflow_revisions: number;
  references: number;
  artifacts: number;
  active_work: number;
  retained_bytes: number;
  reclaimable_bytes: number;
}

export interface RecoveryImpact {
  kind: RecoveryKind;
  subject_id: string;
  revision: string;
  impact_sha256: string;
  counts: RecoveryCounts;
  conflicts: RecoveryConflict[];
  available_actions: RecoveryAction[];
  delete_generated_media: boolean;
  reclaimed_bytes: number;
}

export interface RecoveryItem {
  deletion_id: string;
  kind: RecoveryKind;
  subject_id: string;
  display_label: string;
  original_location: { project_id: string | null; project_label: string | null };
  deleted_at: string;
  purge_after: string;
  state: RecoveryState;
  revision: string;
  counts: RecoveryCounts;
  restore_conflicts: RecoveryConflict[];
  delete_generated_media: boolean;
}

export interface RecoveryPage {
  items: RecoveryItem[];
  next_cursor: string | null;
}
export interface RecoveryCommand {
  expected_revision: string;
  impact_sha256: string;
  operation_key: string;
}
export interface RecoveryResult {
  deletion_id: string;
  kind: RecoveryKind;
  subject_id: string;
  action: RecoveryAction;
  replayed: boolean;
  reclaimed_bytes: number;
}

export type RecoveryBatchAction = "restore" | "purge";
export interface RecoveryBatchSelection {
  deletion_ids: string[];
  action: RecoveryBatchAction;
  restore_unfiled: boolean;
}
export interface RecoveryBatchMember {
  deletion_id: string;
  display_label: string;
  purge_after: string;
  impact: RecoveryImpact;
}
export interface RecoveryBatchPreview {
  batch_id: string;
  revision: string;
  impact_sha256: string;
  action: RecoveryBatchAction;
  policy: "all-or-nothing";
  restore_unfiled: boolean;
  available: boolean;
  expires_at: string;
  items: RecoveryBatchMember[];
}
export interface RecoveryBatchCommand extends RecoveryCommand {
  acknowledgement?: "permanently-delete";
}
export interface RecoveryBatchResult {
  batch_id: string;
  action: RecoveryBatchAction;
  policy: "all-or-nothing";
  reclaimed_bytes: number;
  results: RecoveryResult[];
}
