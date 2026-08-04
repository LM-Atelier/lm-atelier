/** The settings panel and the executor have to agree about what runs. */

import { describe, expect, it } from "vitest";
import { operationForTurn, revisionForTurn, schemaForRevision } from "./turnWorkflow";
import type { Workflow, WorkflowFamily, WorkflowSelection } from "./types";

const LEGACY_SCHEMA = { properties: { legacy: {} } };
const FAMILY_SCHEMA = { properties: { chosen: {} } };

function workflows(): Workflow[] {
  return [
    {
      id: "wf-legacy",
      name: "Legacy image",
      operation: "text_to_image",
      current_revision_id: "rev-legacy",
      revisions: [{ id: "rev-legacy", input_schema_json: LEGACY_SCHEMA }],
    },
    {
      id: "wf-family",
      name: "Chosen image",
      operation: "text_to_image",
      current_revision_id: "rev-family",
      revisions: [{ id: "rev-family", input_schema_json: FAMILY_SCHEMA }],
    },
  ] as unknown as Workflow[];
}

function families(): WorkflowFamily[] {
  return [
    {
      id: "family-1",
      name: "Chosen",
      variants: [
        {
          id: "v1",
          variant_key: "text_to_image",
          name: "Text to image",
          operation: "text_to_image",
          current_revision_id: "rev-family",
          current_revision_version: 2,
          engine: "comfyui",
          capabilities: ["image"],
          trusted: true,
          readiness: "ready",
          readiness_reason: null,
        },
      ],
    },
  ] as unknown as WorkflowFamily[];
}

function selection(mode: WorkflowSelection["mode"], extra: Partial<WorkflowSelection> = {}) {
  return {
    selector_capability: "image",
    mode,
    workflow_family_id: null,
    workflow_revision_id: null,
    legacy_profile_id: null,
    ...extra,
  } as WorkflowSelection;
}

describe("which revision a turn will run", () => {
  it("follows a chosen family over a project's older pin", () => {
    // The defect this exists for: the panel read the legacy pin while the
    // executor honoured the family, so the controls on screen belonged to
    // one workflow and the picture came from another.
    const chosen = revisionForTurn(
      workflows(),
      families(),
      selection("family", { workflow_family_id: "family-1" }),
      "rev-legacy",
      "text_to_image",
    );

    expect(chosen).toBe("rev-family");
    expect(schemaForRevision(workflows(), chosen, "text_to_image")).toEqual(FAMILY_SCHEMA);
  });

  it("keeps a project pinned to an exact revision on that revision", () => {
    expect(
      revisionForTurn(
        workflows(),
        families(),
        selection("revision", { workflow_revision_id: "rev-legacy" }),
        null,
        "text_to_image",
      ),
    ).toBe("rev-legacy");
  });

  it("falls through when the chosen family cannot do this operation", () => {
    // A family with no variant for this turn cannot answer it, so pinning
    // one of its other variants would run the wrong thing confidently.
    expect(
      revisionForTurn(
        workflows(),
        families(),
        selection("family", { workflow_family_id: "family-1" }),
        "rev-legacy",
        "text_to_video",
      ),
    ).toBe("rev-legacy");
  });

  it("lets automatic mean automatic rather than the old pin", () => {
    expect(
      revisionForTurn(workflows(), families(), selection("automatic"), "rev-legacy", "text_to_image"),
    ).toBeUndefined();
  });

  it("still honours a legacy pin when nothing newer was chosen", () => {
    expect(
      revisionForTurn(workflows(), families(), undefined, "rev-legacy", "text_to_image"),
    ).toBe("rev-legacy");
  });

  it("names the operation an attachment implies", () => {
    expect(operationForTurn("image", false)).toBe("text_to_image");
    expect(operationForTurn("image", true)).toBe("image_to_image");
    expect(operationForTurn("video", true)).toBe("image_to_video");
  });
});
