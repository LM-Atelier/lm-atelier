import { describe, expect, it } from "vitest";
import { studioStepAncestors, studioStepOrigin } from "./studioStepOrigin";
import type { StudioStep } from "./useStudioSession";

function step(artifactId: string, beforeArtifactId: string | null): StudioStep {
  return {
    messageId: `message-${artifactId}`,
    artifactId,
    instruction: beforeArtifactId ? `made from ${beforeArtifactId}` : "",
    beforeArtifactId,
    isSource: beforeArtifactId === null,
    generationIdentity: null,
  };
}

describe("where a Studio result came from", () => {
  it("says nothing for results each made from the step before them", () => {
    const steps = [step("a", null), step("b", "a"), step("c", "b")];

    expect(steps.map((_, index) => studioStepOrigin(steps, index))).toEqual([null, null, null]);
  });

  it("names an earlier step a result was made from, and the original by name", () => {
    const steps = [step("a", null), step("b", "a"), step("c", "b"), step("d", "b"), step("e", "a")];

    expect(steps.map((_, index) => studioStepOrigin(steps, index))).toEqual([
      null,
      null,
      null,
      "From step 1",
      "From the original",
    ]);
  });

  it("says nothing for a result whose picture is not in the strip", () => {
    const steps = [step("a", null), step("b", "a"), step("c", "elsewhere")];

    expect(studioStepOrigin(steps, 2)).toBeNull();
  });

  it("takes the nearest earlier appearance of a picture that appears twice", () => {
    // An edit that gave back exactly the picture it was given.
    const steps = [step("a", null), step("b", "a"), step("a", "b"), step("c", "a")];

    expect(studioStepOrigin(steps, 3)).toBeNull();
  });
});

describe("the results a Studio result was made from", () => {
  it("runs back through each picture it was made from to the original, and no further", () => {
    // Two in a row, a branch from the first, then one more on the branch.
    const steps = [step("a", null), step("b", "a"), step("c", "b"), step("d", "b"), step("e", "d")];

    expect([...studioStepAncestors(steps, 4)].sort()).toEqual([0, 1, 3]);
    expect([...studioStepAncestors(steps, 2)].sort()).toEqual([0, 1]);
    expect([...studioStepAncestors(steps, 0)]).toEqual([]);
  });

  it("stops at a picture that is not in the strip", () => {
    const steps = [step("a", null), step("b", "elsewhere"), step("c", "b")];

    expect([...studioStepAncestors(steps, 2)]).toEqual([1]);
  });
});
