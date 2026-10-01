/** What the Studio's apply button says, tool by tool. */

import { describe, expect, it } from "vitest";
import { studioApplyLabel } from "./studioApplyPlan";
import { enlargementIdentity } from "./studioEnlargement";
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

  it("names how much Enhance enlarges only when the workflow's graph proves it", () => {
    expect(studioApplyLabel(tools("enhance"), false, 0)).toBe("Enlarge");
    const range = { key: "upscale_factor", label: "", type: "integer" as const, default: 2, minimum: 1, maximum: 4, step: 1,
      choices: [], scope: "workflow" as const, visibility: "basic" as const, restart_required: false, available: true,
      unavailable_reason: null, help: "" };
    const answer = { version: 1 as const, status: "ready" as const, workflow_revision_id: "rev", factor: range,
      fixed_factor: null, request_authorized: false as const };
    // A factor the person can change sets the workflow's scale, which another
    // step may multiply, so neither its default nor a chosen one is named as the size.
    expect(studioApplyLabel(tools("enhance"), false, 0, answer)).toBe("Enlarge");
    const chosen = tools("enhance", { upscaleChoice: { preview: enlargementIdentity(answer), factor: 4 } });
    expect(studioApplyLabel(chosen, false, 0, answer)).toBe("Enlarge");
    expect(studioApplyLabel(tools("enhance"), false, 0, { ...answer, factor: null, fixed_factor: 2 })).toBe("Enlarge 2x");
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
