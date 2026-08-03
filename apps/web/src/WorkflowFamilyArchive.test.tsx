/** Archiving says what survives, because "archive" invites the opposite guess. */

import { describe, expect, it } from "vitest";
import { needsAttention, survivingWork } from "./workflowFamilyImpact";
import type { WorkflowFamilyRemovalImpact } from "./types";

function impact(overrides: Partial<WorkflowFamilyRemovalImpact> = {}): WorkflowFamilyRemovalImpact {
  return {
    family_id: "family-1",
    removal_strategy: "archive",
    archive_blocked: false,
    revision_count: 0,
    current_revision_count: 0,
    chat_selection_count: 0,
    project_selection_count: 0,
    project_revision_pin_count: 0,
    active_run_count: 0,
    queued_step_count: 0,
    historical_run_count: 0,
    active_activation_count: 0,
    default_for: [],
    dependencies: [],
    ...overrides,
  };
}

describe("what archiving a family costs", () => {
  it("names the work that keeps running", () => {
    const kept = survivingWork(
      impact({
        queued_step_count: 2,
        active_run_count: 1,
        historical_run_count: 40,
        project_revision_pin_count: 1,
        revision_count: 6,
      }),
    );

    // Nothing is deleted, so the reassuring half has to be said out loud -
    // otherwise "archive" reads as "throw away" and nobody presses it.
    expect(kept).toEqual([
      "2 queued steps still runs",
      "1 run in progress finishes",
      "40 past runs stay in their chats",
      "1 project pinned to an exact revision keeps working",
      "6 saved revisions are kept",
    ]);
  });

  it("names what the user has to deal with afterwards", () => {
    const attention = needsAttention(
      impact({
        chat_selection_count: 3,
        project_selection_count: 1,
        default_for: ["image", "video"],
        dependencies: [
          {
            resource_kind: "model_install",
            resource_id: "m1",
            resource_name: "Only here",
            binding_count: 1,
            revision_count: 1,
            current_revision: true,
            shared: false,
            other_workflow_count: 0,
            other_family_ids: [],
          },
          {
            resource_kind: "model_install",
            resource_id: "m2",
            resource_name: "Used elsewhere too",
            binding_count: 1,
            revision_count: 1,
            current_revision: true,
            shared: true,
            other_workflow_count: 2,
            other_family_ids: ["family-2"],
          },
        ],
      }),
    );

    expect(attention).toEqual([
      "4 chats and projects choosing it fall back to automatic",
      "It is the default for image, video",
      // Only the exclusive one is counted. A shared file is the reassuring
      // case, and nothing is being deleted either way - what matters is
      // which downloads nothing else would be using.
      "1 model file would then be used by nothing else",
    ]);
  });

  it("says nothing at all about a family nothing depends on", () => {
    expect(survivingWork(impact())).toEqual([]);
    expect(needsAttention(impact())).toEqual([]);
  });
});
