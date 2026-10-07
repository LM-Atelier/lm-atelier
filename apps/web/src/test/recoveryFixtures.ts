import type { RecoveryCommand, RecoveryImpact, RecoveryItem } from "../recoveryTypes";

export const recoveryCommand: RecoveryCommand = {
  expected_revision: "b".repeat(64), impact_sha256: "c".repeat(64), operation_key: "trash-garden",
};
export function recoveryItem(subjectId = "chat-1"): RecoveryItem {
  return {
    deletion_id: `deleted-${subjectId}`, kind: "chat", subject_id: subjectId, display_label: "Garden notes",
    original_location: { project_id: null, project_label: null },
    deleted_at: "2026-10-02T00:00:00Z", purge_after: "2099-11-01T00:00:00Z", state: "recoverable",
    revision: "a".repeat(64), restore_conflicts: [], delete_generated_media: false,
    counts: { chats: 0, messages: 0, message_parts: 0, response_revisions: 0, runs: 0, jobs: 0, work_plans: 0,
      workflow_families: 0, workflow_definitions: 0, workflow_revisions: 0,
      references: 0, artifacts: 0, active_work: 0, retained_bytes: 0, reclaimable_bytes: 0 },
  };
}
export function recoveryImpact(subjectId = "chat-1", actions: RecoveryImpact["available_actions"] = ["trash"]): RecoveryImpact {
  return { kind: "chat", subject_id: subjectId, revision: recoveryCommand.expected_revision,
    impact_sha256: recoveryCommand.impact_sha256, counts: recoveryItem(subjectId).counts,
    conflicts: [], available_actions: actions, delete_generated_media: false, reclaimed_bytes: 0 };
}
