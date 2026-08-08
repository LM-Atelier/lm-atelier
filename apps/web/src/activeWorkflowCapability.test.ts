import { describe, expect, it } from "vitest";
import {
  activeWorkflowCapability,
  variantServesComposerCapability,
  workflowChoiceKind,
} from "./activeWorkflowCapability";
import type {
  WorkflowFamilyVariant,
  WorkflowSelection,
  WorkflowSelectionMode,
} from "./types";

function selection(mode: WorkflowSelectionMode): WorkflowSelection {
  return {
    selector_capability: "image",
    mode,
    workflow_family_id: mode === "family" ? "family-1" : null,
    workflow_revision_id: mode === "revision" ? "revision-1" : null,
    legacy_profile_id: mode === "legacy" ? "profile-1" : null,
  };
}

describe("activeWorkflowCapability", () => {
  it.each([
    ["text", "chat"],
    ["image", "image"],
    ["video", "video"],
    ["auto", null],
  ] as const)("maps %s without inventing a vision capability", (mode, expected) => {
    expect(activeWorkflowCapability(mode)).toBe(expected);
  });
});

describe("workflowChoiceKind", () => {
  it("treats an absent row as the inherited default", () => {
    expect(workflowChoiceKind(undefined)).toBe("default");
  });

  it.each([
    ["default", "default"],
    ["inherit", "default"],
    ["automatic", "automatic"],
    ["family", "explicit"],
    ["legacy", "compatibility"],
    ["revision", "compatibility"],
  ] as const)("presents %s as %s", (mode, expected) => {
    expect(workflowChoiceKind(selection(mode))).toBe(expected);
  });
});

describe("variantServesComposerCapability", () => {
  const variant: WorkflowFamilyVariant = {
    id: "variant-1",
    variant_key: "create",
    name: "Create",
    operation: "text_to_image",
    current_revision_id: "revision-1",
    current_revision_version: 1,
    engine: "comfyui",
    capabilities: [],
    trusted: true,
    readiness: "ready",
    readiness_reason: null,
  };

  it("uses the declared capability when one is present", () => {
    expect(variantServesComposerCapability(
      { ...variant, operation: "custom", capabilities: ["video"] },
      "video",
    )).toBe(true);
  });

  it.each([
    ["text", "chat", true],
    ["text_to_image", "image", true],
    ["image_to_image", "image", true],
    ["text_to_video", "video", true],
    ["image_to_video", "video", true],
    ["text_to_video", "image", false],
  ] as const)("maps operation %s to %s=%s", (operation, capability, expected) => {
    expect(variantServesComposerCapability(
      { ...variant, operation },
      capability,
    )).toBe(expected);
  });
});
