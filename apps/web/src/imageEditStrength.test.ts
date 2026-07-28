import { describe, expect, it } from "vitest";
import { estimateImageEditStrength } from "./imageEditStrength";

describe("estimateImageEditStrength", () => {
  it.each([
    ["Slightly brighten the lighting and keep everything else the same.", "minimal", 0.38],
    ["Make it green.", "localized", 0.50],
    ["Give the person a new formal outfit without changing their face.", "replacement", 0.66],
    ["Replace the background with a moonlit city scene.", "replacement", 0.66],
    ["Restyle the entire image as a watercolor painting.", "global", 0.82],
    ["Make this better.", "fallback", 0.56],
    ["Do not change the clothing; just brighten the lighting.", "minimal", 0.38],
  ])("classifies %s", (prompt, scope, value) => {
    expect(estimateImageEditStrength(prompt)).toEqual({
      scope,
      value,
      confidence: scope === "localized" ? "medium" : scope === "fallback" ? "low" : "high",
    });
  });

  it("clamps the estimate to workflow bounds", () => {
    expect(estimateImageEditStrength("Restyle the image", 0.6, 0.7).value).toBe(0.7);
  });
});