import type { Workflow, WorkflowFamily, WorkflowSelection } from "./types";

export const DEFAULT_CHAT_WORKFLOW_SELECTIONS: WorkflowSelection[] = [
  { selector_capability: "image", mode: "default", workflow_family_id: null,
    workflow_revision_id: null, legacy_profile_id: null },
  { selector_capability: "video", mode: "default", workflow_family_id: null,
    workflow_revision_id: null, legacy_profile_id: null },
];

export const DEFAULT_PROJECT_WORKFLOW_SELECTIONS: WorkflowSelection[] = [
  { selector_capability: "image", mode: "inherit", workflow_family_id: null,
    workflow_revision_id: null, legacy_profile_id: null },
  { selector_capability: "video", mode: "inherit", workflow_family_id: null,
    workflow_revision_id: null, legacy_profile_id: null },
];

/** Give older App fixtures the deterministic workspace family current APIs return. */
export function familiesForWorkflows(workflows: Workflow[]): WorkflowFamily[] {
  return (["image", "video"] as const).flatMap((capability) => {
    const variants = workflows
      .filter((workflow) => workflow.operation.includes(capability))
      .map((workflow) => {
        const revision = workflow.revisions.find((item) => item.id === workflow.current_revision_id);
        return {
          id: workflow.id,
          variant_key: workflow.operation,
          name: workflow.name,
          operation: workflow.operation,
          current_revision_id: revision?.id ?? null,
          current_revision_version: revision?.version ?? null,
          engine: revision?.engine ?? null,
          capabilities: [capability],
          trusted: revision?.trusted ?? false,
          readiness: "ready" as const,
          readiness_reason: null,
        };
      });
    if (variants.length === 0) return [];
    return [{
      id: `test-${capability}-family`,
      name: `Test ${capability} family`,
      description: "",
      use_case: capability,
      tags: [],
      enabled: true,
      archived: false,
      compatibility: false,
      variants,
      preferences: [
        { selector_capability: capability, enabled: true, is_default: true, sort_order: 0 },
      ],
      created_at: "2026-08-03T00:00:00Z",
      updated_at: "2026-08-03T00:00:00Z",
    }];
  });
}
