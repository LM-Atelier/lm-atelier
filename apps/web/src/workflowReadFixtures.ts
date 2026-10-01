import { fixtureFamilyForWorkflow, resetWorkflowFamilyFixtures } from "./workflowFamilyReadFixtures";
import { vi } from "vitest";
import { api } from "./api";
import type { Workflow } from "./types";

// Existing interaction tests share neutral workflow fixtures. Feed those
// fixtures to both new reads; transport behavior has separate client tests.
export function mockWorkflowReadsFromFixture(load: () => Promise<Workflow[]>) {
  resetWorkflowFamilyFixtures();
  vi.mocked(api.workflowSummaries).mockImplementation(async (options = {}) => {
    const rows = await Promise.all((await load()).map(async workflow => ({
    id: workflow.id, family_id: workflow.family_id === undefined ? await fixtureFamilyForWorkflow(workflow.id) ?? null : workflow.family_id, name: workflow.name,
    operation: workflow.operation, description: workflow.description,
    current_revision_id: workflow.current_revision_id, revision_count: workflow.revisions.length,
    created_at: "2026-09-01T00:00:00Z", updated_at: "2026-09-01T00:00:00Z",
    })));
    const needle = options.search?.trim().toLocaleLowerCase() ?? "";
    return rows.filter(row => (!options.ungroupedOnly || row.family_id === null)
      && (!options.workflowIds || options.workflowIds.includes(row.id))
      && (!options.operation || row.operation === options.operation)
      && (!needle || [row.name, row.description].some(value => value.toLocaleLowerCase().includes(needle))))
      .slice(options.offset ?? 0, options.limit === undefined ? undefined : (options.offset ?? 0) + options.limit);
  });
  vi.mocked(api.workflow).mockImplementation(async (id) => {
    const workflow = (await load()).find((value) => value.id === id);
    if (!workflow) throw new Error("Workflow fixture not found.");
    return workflow;
  });
}

export function mockWorkflowConsumerReadsFromFixture(load: () => Promise<Workflow[]>) {
  vi.mocked(api.workflowRevisionChoices).mockImplementation(async () =>
    (await load()).flatMap((workflow) => workflow.revisions.map((revision) => ({
      revision_id: revision.id, workflow_id: workflow.id, workflow_name: workflow.name,
      operation: workflow.operation, version: revision.version,
    }))));
  vi.mocked(api.workflowRevisionSchema).mockImplementation(async (id) => {
    for (const workflow of await load()) {
      const revision = workflow.revisions.find((value) => value.id === id);
      if (revision) return { revision_id: id, workflow_id: workflow.id,
        operation: workflow.operation, input_schema_json: revision.input_schema_json };
    }
    throw new Error("Workflow revision fixture not found.");
  });
}

export { mockWorkflowFamilyPages } from "./workflowFamilyReadFixtures";
