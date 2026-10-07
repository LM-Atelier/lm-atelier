import { describe, expect, it } from "vitest";
import { EDIT_STRENGTH_NOTE, editStrengthNote } from "./editStrengthNote";

const automatic = (scope: string) => ({
  image_edit: {
    strength: { mode: "auto", parameter: "denoise", value: 0.5, scope, reason_codes: [] },
  },
});

const reviewed = (assessment: Record<string, unknown>, reason?: string) => ({
  image_edit_verification: {
    status: "complete",
    automatic_retry_executed: false,
    ...(reason ? { reason } : {}),
    assessment: { unrelated_content_preserved: true, ...assessment },
  },
});

describe("editStrengthNote", () => {
  it("says a specific change was redrawn at an automatic strength and names the control", () => {
    for (const scope of ["minimal", "localized", "replacement", "fallback"]) {
      expect(editStrengthNote(automatic(scope))).toBe(EDIT_STRENGTH_NOTE);
    }
    expect(EDIT_STRENGTH_NOTE).toContain("set Change strength to Manual");
  });

  it("stays quiet for a change to the whole picture, which a redraw does well", () => {
    expect(editStrengthNote(automatic("global"))).toBeNull();
  });

  it("stays quiet when the person set the strength", () => {
    expect(editStrengthNote({
      image_edit: { strength: { mode: "manual", parameter: "denoise", value: 0.7, reason_codes: [] } },
    })).toBeNull();
  });

  it("stays quiet for an edit that reads the words beside the picture", () => {
    expect(editStrengthNote({
      image_edit: { default_change_strength_applied: false, policy: "preserve_unrequested_details_v1" },
    })).toBeNull();
    expect(editStrengthNote(undefined)).toBeNull();
  });

  it("stays quiet once the edit review found the change", () => {
    expect(editStrengthNote({
      ...automatic("localized"),
      ...reviewed({ requested_change_visible: true }),
    })).toBeNull();
  });

  it("keeps the line when the review did not find the change, or measured none", () => {
    expect(editStrengthNote({
      ...automatic("localized"),
      ...reviewed({ requested_change_visible: false }),
    })).toBe(EDIT_STRENGTH_NOTE);
    expect(editStrengthNote({
      ...automatic("localized"),
      ...reviewed({ requested_change_visible: true }, "no_measurable_change"),
    })).toBe(EDIT_STRENGTH_NOTE);
  });
});
