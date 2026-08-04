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
      preferences: [],
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
  it("follows a chosen chat family over a project's family", () => {
    const chosen = revisionForTurn(
      families(),
      "image",
      selection("family", { workflow_family_id: "family-1" }),
      selection("revision", { workflow_revision_id: "rev-legacy" }),
      "text_to_image",
    );

    expect(chosen).toBe("rev-family");
    expect(schemaForRevision(workflows(), chosen, "text_to_image")).toEqual(FAMILY_SCHEMA);
  });

  it("lets a default chat inherit an exact project revision", () => {
    expect(
      revisionForTurn(
        families(),
        "image",
        selection("default"),
        selection("revision", { workflow_revision_id: "rev-legacy" }),
        "text_to_image",
      ),
    ).toBe("rev-legacy");
  });

  it("lets a default chat inherit the project's family variant", () => {
    expect(
      revisionForTurn(
        families(),
        "image",
        selection("default"),
        selection("family", { workflow_family_id: "family-1" }),
        "text_to_image",
      ),
    ).toBe("rev-family");
  });

  it("does not invent controls when the chosen family lacks the operation", () => {
    expect(
      revisionForTurn(
        families(),
        "video",
        selection("family", { workflow_family_id: "family-1" }),
        selection("revision", { workflow_revision_id: "rev-legacy" }),
        "text_to_video",
      ),
    ).toBeNull();
  });

  it("does not invent controls for an ambiguous family operation", () => {
    const ambiguous = families();
    ambiguous[0].variants.push({ ...ambiguous[0].variants[0], id: "v2" });
    expect(
      revisionForTurn(
        ambiguous,
        "image",
        selection("family", { workflow_family_id: "family-1" }),
        null,
        "text_to_image",
      ),
    ).toBeNull();
  });

  it("keeps automatic and legacy profile choices unresolved", () => {
    expect(
      revisionForTurn(
        families(),
        "image",
        selection("automatic"),
        selection("revision", { workflow_revision_id: "rev-legacy" }),
        "text_to_image",
      ),
    ).toBeNull();
    expect(
      revisionForTurn(
        families(),
        "image",
        selection("legacy", { legacy_profile_id: "profile-1" }),
        selection("revision", { workflow_revision_id: "rev-legacy" }),
        "text_to_image",
      ),
    ).toBeNull();
  });

  it("uses the deterministic workspace default after both scopes inherit", () => {
    const available = families();
    available[0].preferences = [
      { selector_capability: "image", enabled: true, is_default: true, sort_order: 0 },
    ];
    expect(
      revisionForTurn(
        available,
        "image",
        selection("default"),
        selection("inherit"),
        "text_to_image",
      ),
    ).toBe("rev-family");
  });

  it("fails closed until required selection responses load", () => {
    expect(
      revisionForTurn(families(), "image", undefined, null, "text_to_image"),
    ).toBeNull();
    expect(
      revisionForTurn(
        families(),
        "image",
        selection("default"),
        undefined,
        "text_to_image",
      ),
    ).toBeNull();
  });

  it("does not substitute the first workflow for an unresolved selection", () => {
    expect(schemaForRevision(workflows(), null, "text_to_image")).toBeUndefined();
  });

  it("names the operation an attachment implies", () => {
    expect(operationForTurn("image", false)).toBe("text_to_image");
    expect(operationForTurn("image", true)).toBe("image_to_image");
    expect(operationForTurn("video", true)).toBe("image_to_video");
  });
});
