import type { WorkflowFamily, WorkflowReadyRevision } from "./types";
import type { WorkflowReadPageOptions } from "./workflowReadQuery";

export function readyWorkflowPage(families: WorkflowFamily[], options: WorkflowReadPageOptions = {}) {
  const rows: WorkflowReadyRevision[] = [];
  const search = (options.search ?? "").toLowerCase();
  for (const family of families) {
    if (!family.enabled || family.archived
      || !family.preferences.some(value => value.selector_capability === "image" && value.enabled)) continue;
    for (const variant of family.variants) {
      const id = variant.current_revision_id;
      if (!id || !variant.current_revision_version || variant.readiness !== "ready"
        || variant.operation !== (options.operation ?? "text_to_image")) continue;
      if (options.revisionIds?.length && !options.revisionIds.includes(id)) continue;
      if (search && !`${family.name} ${variant.name}`.toLowerCase().includes(search)) continue;
      if (rows.some(row => row.revision_id === id)) continue;
      rows.push({ family_id: family.id, family_name: family.name, workflow_id: variant.id,
        workflow_name: variant.name, revision_id: id, revision_version: variant.current_revision_version,
        operation: variant.operation });
    }
  }
  const offset = options.offset ?? 0;
  return rows.slice(offset, offset + (options.limit ?? 50));
}
