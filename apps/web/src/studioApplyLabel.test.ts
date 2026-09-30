/** What the Studio's apply button says, tool by tool. */

import { describe, expect, it } from "vitest";
import { studioApplyLabel } from "./studioApplyPlan";
import { initialToolState, type StudioToolState } from "./studioToolState";

function tools(kind: StudioToolState["kind"], changes: Partial<StudioToolState> = {}): StudioToolState {
  return { ...initialToolState(), kind, ...changes };
}

describe("the apply button's words", () => {
  it.each([
    ["extend", "Extend"],
    ["text", "Replace words"],
    ["remove", "Remove"],
    ["relight", "Relight"],
    ["isolate", "Cut out"],
    ["background", "Replace background"],
    ["subject", "Replace subject"],
    ["instruct", "Apply edit"],
  ] as const)("names the %s tool's own action", (kind, label) => {
    expect(studioApplyLabel(tools(kind), false, 0)).toBe(label);
  });

  it("says how much Enhance enlarges", () => {
    expect(studioApplyLabel(tools("enhance", { upscaleFactor: 4 }), false, 0)).toBe("Enlarge 4x");
  });

  it("applies to the selection once a selecting tool has marked something, and not before", () => {
    expect(studioApplyLabel(tools("brush"), false, 0.2)).toBe("Apply to selection");
    expect(studioApplyLabel(tools("brush"), false, 0)).toBe("Apply edit");
    // Instruct edits the whole picture whatever is marked.
    expect(studioApplyLabel(tools("instruct"), false, 0.2)).toBe("Apply edit");
  });

  it("says it is applying while an edit arrives, whatever the tool", () => {
    expect(studioApplyLabel(tools("extend"), true, 0)).toBe("Applying…");
    expect(studioApplyLabel(tools("brush"), true, 0.2)).toBe("Applying…");
  });
});
