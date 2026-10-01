/** The factor an enlargement sends: only one the workflow applies, and only as it takes it. */

import { describe, expect, it } from "vitest";
import {
  chosenFactor,
  describeFixedEnlargement,
  enlargementChoices,
  enlargementFactor,
  enlargementIdentity,
  enlargementScale,
  type EnlargementPreview,
} from "./studioEnlargement";
import { buildTurnRequest } from "./turnRequest";
import type { SettingField } from "./types";

function field(overrides: Partial<SettingField> = {}): SettingField {
  return {
    key: "upscale_factor",
    label: "Scale",
    type: "number",
    default: 3,
    minimum: 1,
    maximum: 4,
    step: 0.5,
    choices: [],
    scope: "workflow",
    visibility: "basic",
    restart_required: false,
    available: true,
    unavailable_reason: null,
    help: "",
    ...overrides,
  };
}

function preview(overrides: Partial<EnlargementPreview> = {}): EnlargementPreview {
  return {
    version: 1,
    status: "ready",
    workflow_revision_id: "rev-enlarge",
    factor: null,
    fixed_factor: null,
    request_authorized: false,
    ...overrides,
  };
}

describe("the factor an enlargement sends", () => {
  it("is the workflow's own default until the person chooses", () => {
    expect(enlargementFactor(field(), null)).toBe(3);
    expect(enlargementFactor(field(), 2)).toBe(2);
    // A field with neither a default nor a choice sends nothing: the workflow's own value stands.
    expect(enlargementFactor(field({ default: null }), null)).toBeNull();
  });

  it("keeps to the bounds the field declares, and invents none it does not", () => {
    expect(enlargementFactor(field(), 9)).toBe(4);
    expect(enlargementFactor(field(), 0.2)).toBe(1);
    expect(enlargementFactor(field({ minimum: null, maximum: null }), 12)).toBe(12);
    expect(enlargementFactor(field({ maximum: null }), 50)).toBe(50);
    expect(enlargementFactor(field({ minimum: null }), 0.25)).toBe(0.25);
  });

  it("keeps to the field's multiple counted from zero, as the server checks it", () => {
    const tenths = field({ minimum: 0.3, maximum: 2, multiple_of: 0.2, default: 0.4, step: null });

    expect(enlargementFactor(tenths, null)).toBe(0.4);
    expect(enlargementFactor(tenths, 0.5)).toBe(0.6);
    // Not 0.3 itself, which is no multiple of 0.2, but the first multiple above it.
    expect(enlargementFactor(tenths, 0.3)).toBe(0.4);
    expect(enlargementFactor({ ...tenths, maximum: 1.9 }, 1.9)).toBe(1.8);
    for (const chosen of [0.31, 0.77, 1.05, 1.99]) {
      const sent = enlargementFactor(tenths, chosen)!;
      expect(Math.abs(sent / 0.2 - Math.round(sent / 0.2))).toBeLessThan(1e-9);
    }
  });

  it("keeps a whole-number field whole", () => {
    expect(enlargementFactor(field({ type: "integer", step: 1 }), 2.6)).toBe(3);
    expect(enlargementFactor(field({ type: "integer", step: null, minimum: 1.5, maximum: 3.5 }), 1.6)).toBe(2);
  });

  it("keeps a whole-number field to whole numbers that are also its multiple", () => {
    // Whole multiples of 1.5 are the multiples of three: 4 is nearer 3 than 6, and 5 nearer 6.
    const halves = field({ type: "integer", minimum: 1, maximum: 8, multiple_of: 1.5, default: 3, step: null });
    expect(enlargementFactor(halves, 4)).toBe(3);
    expect(enlargementFactor(halves, 5)).toBe(6);
    expect(enlargementFactor(halves, 8)).toBe(6);
    // No whole number from one to two is a multiple of 0.3, so there is nothing to send.
    const none = field({ type: "integer", minimum: 1, maximum: 2, multiple_of: 0.3, default: 1.5, step: null });
    expect(enlargementFactor(none, null)).toBeNull();
    expect(enlargementFactor(none, 2)).toBeNull();
    // Every whole number is a multiple of a quarter.
    expect(enlargementFactor(field({ type: "integer", multiple_of: 0.25, step: null }), 2.4)).toBe(2);
  });

  it("offers a control that steps only through factors the workflow takes", () => {
    const halves = field({ type: "integer", minimum: 1, maximum: 8, multiple_of: 1.5, step: 1.5 });
    expect(enlargementScale(halves)).toEqual({ minimum: 3, maximum: 6, step: 3 });
    // From the first multiple of 0.2 above 0.3, not from 0.3 itself.
    expect(enlargementScale(field({ minimum: 0.3, maximum: 2, multiple_of: 0.2, step: 0.1 }))).toEqual({
      minimum: 0.4,
      maximum: 2,
      step: 0.2,
    });
    expect(enlargementScale(field({ type: "integer", minimum: 1, maximum: 2, multiple_of: 0.3 }))).toBeNull();
    // With no multiple, the field's own step stays the control's where it suits, and open bounds stay open.
    expect(enlargementScale(field())).toEqual({ minimum: 1, maximum: 4, step: 0.5 });
    expect(enlargementScale(field({ type: "integer", step: 0.5 }))).toEqual({ minimum: 1, maximum: 4, step: 1 });
    expect(enlargementScale(field({ minimum: null, maximum: null, step: null }))).toEqual({
      minimum: null,
      maximum: null,
      step: "any",
    });
  });

  it("keeps a chosen factor from a list, and otherwise the list's own default", () => {
    const listed = field({ type: "enum", choices: [4, 2, "8"], default: 4, minimum: null, maximum: null, step: null });

    expect(enlargementChoices(listed)).toEqual([2, 4]);
    expect(enlargementFactor(listed, 2)).toBe(2);
    expect(enlargementFactor(listed, 3)).toBe(4);
    expect(enlargementFactor({ ...listed, default: 8 }, null)).toBe(2);
  });

  it("sends none when the workflow offers no factor to choose", () => {
    expect(enlargementFactor(null, 2)).toBeNull();
    expect(enlargementFactor(field({ available: false }), 2)).toBeNull();
    expect(enlargementChoices(null)).toBeNull();
    expect(enlargementChoices(field())).toBeNull();
  });

  it("holds a choice only for the preview it was made under", () => {
    const shown = preview({ factor: field() });
    const choice = { preview: enlargementIdentity(shown), factor: 2 };

    expect(chosenFactor(shown, choice)).toBe(2);
    // A recipe that changes the default, or another workflow, starts again from its own default.
    expect(chosenFactor(preview({ factor: field({ default: 2.5 }) }), choice)).toBeNull();
    expect(chosenFactor(preview({ factor: field(), workflow_revision_id: "rev-other" }), choice)).toBeNull();
    expect(chosenFactor(shown, null)).toBeNull();
  });

  it("says how much a workflow that sets its own size enlarges, only when its graph shows it", () => {
    expect(describeFixedEnlargement(preview({ fixed_factor: 4 }))).toBe("Enlarges 4x, set by the workflow.");
    expect(describeFixedEnlargement(preview())).toBe("The workflow sets how much it enlarges.");
  });
});

describe("a turn said as an enlargement", () => {
  const base = { text: "Enlarge the picture and restore its detail", mode: "image" as const, inputArtifactIds: ["art-1"], settings: {} };

  it("says so only when it is one, so every other turn reads as it always did", () => {
    expect(buildTurnRequest({ ...base, upscale: true }).upscale).toBe(true);
    expect(buildTurnRequest(base)).not.toHaveProperty("upscale");
    expect(buildTurnRequest({ ...base, upscale: false })).not.toHaveProperty("upscale");
  });
});
