import { describe, expect, it } from "vitest";
import {
  calibratedImageEditStrength,
  estimateImageEditStrength,
  workflowImageEditCalibration,
} from "./imageEditStrength";

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

describe("workflow image edit calibration", () => {
  const schema = {
    type: "object",
    properties: {
      strength: { type: "number", default: 0.9 },
      steps: { type: "integer", default: 4 },
    },
    "x-lm-atelier-edit-calibration": {
      version: 1,
      edit_strength: {
        parameter: "strength",
        minimum: 0,
        maximum: 1,
        recommended: {
          minimal: 0.3,
          localized: 0.45,
          replacement: 0.6,
          global: 0.8,
          fallback: 0.5,
        },
      },
      schedule: {
        steps_parameter: "steps",
        minimum_effective_steps: {
          localized: 2,
          replacement: 3,
          global: 3,
        },
      },
    },
  };

  it("reads a custom strength parameter and applies the short-step budget", () => {
    const calibration = workflowImageEditCalibration(schema);

    expect(calibration?.parameter).toBe("strength");
    expect(calibratedImageEditStrength("Replace the jacket", calibration, 4)).toEqual({
      scope: "replacement",
      value: 0.75,
      confidence: "high",
    });
  });

  it("ignores malformed optional contracts", () => {
    expect(workflowImageEditCalibration({
      ...schema,
      "x-lm-atelier-edit-calibration": {
        ...schema["x-lm-atelier-edit-calibration"],
        version: 2,
      },
    })).toBeNull();
  });
});