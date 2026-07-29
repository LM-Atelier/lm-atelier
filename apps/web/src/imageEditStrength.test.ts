import { describe, expect, it } from "vitest";
import {
  calibratedImageEditStrength,
  estimateImageEditStrength,
  resolveImageEditStrengthMode,
  workflowImageEditCalibration,
} from "./imageEditStrength";

describe("estimateImageEditStrength", () => {
  it.each([
    ["Slightly brighten the lighting and keep everything else the same.", "minimal", 0.38, "high"],
    ["Make it green.", "localized", 0.50, "medium"],
    ["Give the person a new formal outfit without changing their face.", "replacement", 0.66, "high"],
    ["Replace the background with a moonlit city scene.", "replacement", 0.66, "high"],
    ["Replace the mannequin's red sweatshirt with a royal-blue blazer. Keep the mannequin's head, pose, hands, pants, framing, lighting, and background unchanged.", "replacement", 0.66, "high"],
    ["Make her top red.", "localized", 0.50, "high"],
    ["Recolor the second person's jacket orange.", "localized", 0.50, "high"],
    ["Change only the rightmost person into a marble statue.", "replacement", 0.66, "high"],
    ["Increase the brightness.", "minimal", 0.38, "high"],
    ["Correct the harsh green color cast and brighten the foreground subjects.", "minimal", 0.38, "high"],
    ["Make the car blue.", "localized", 0.50, "high"],
    ["Give him a short beard.", "replacement", 0.66, "high"],
    ["Make her smile.", "replacement", 0.66, "high"],
    ["Straighten the horizon.", "replacement", 0.66, "high"],
    ["Convert it to black and white.", "global", 0.82, "high"],
    ["Extend the canvas to the left.", "global", 0.82, "high"],
    ["Relight the scene from the left.", "global", 0.82, "high"],
    ["Correct the white balance.", "minimal", 0.38, "high"],
    ["Desaturate everything except the umbrella.", "localized", 0.50, "high"],
    ["Change the center flower into a sunflower.", "replacement", 0.66, "high"],
    ["Restyle the entire image as a watercolor painting.", "global", 0.82, "high"],
    ["Make this better.", "fallback", 0.56, "low"],
    ["Do not change the clothing; just brighten the lighting.", "minimal", 0.38, "high"],
  ])("classifies %s", (prompt, scope, value, confidence) => {
    expect(estimateImageEditStrength(prompt)).toEqual({
      scope,
      value,
      confidence,
    });
  });

  it("clamps the estimate to workflow bounds", () => {
    expect(estimateImageEditStrength("Restyle the image", 0.6, 0.7).value).toBe(0.7);
  });
});

describe("image edit strength mode", () => {
  it("keeps profile and default-preset numeric defaults on Auto", () => {
    expect(resolveImageEditStrengthMode(
      "denoise",
      [{ denoise: 0.41 }, { denoise: 0.55 }, undefined, {}, undefined, {}],
      [false, false, true, true, true, true],
    )).toBe("auto");
  });

  it("preserves legacy user numeric choices and explicit mode precedence", () => {
    const numericManualLayers = [false, false, true, true, true, true];
    expect(resolveImageEditStrengthMode(
      "denoise",
      [{ _image_edit_strength_mode: "auto" }, {}, undefined, { denoise: 0.62 }, undefined, {}],
      numericManualLayers,
    )).toBe("manual");
    expect(resolveImageEditStrengthMode(
      "denoise",
      [{ denoise: 0.41 }, {}, undefined, { denoise: 0.62 }, undefined, { _image_edit_strength_mode: "auto" }],
      numericManualLayers,
    )).toBe("auto");
  });
});

describe("workflow image edit calibration", () => {
  const schema = {
    type: "object",
    properties: {
      strength: { type: "number", default: 0.9 },
      steps: { type: "integer", default: 8 },
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
          replacement: 7.2,
          global: 3,
        },
      },
    },
  };

  it("reads a custom strength parameter and applies the short-step budget", () => {
    const calibration = workflowImageEditCalibration(schema);

    expect(calibration?.parameter).toBe("strength");
    expect(calibratedImageEditStrength("Replace the jacket", calibration, 8)).toEqual({
      scope: "replacement",
      value: 0.9,
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
    expect(workflowImageEditCalibration({
      ...schema,
      "x-lm-atelier-edit-calibration": {
        ...schema["x-lm-atelier-edit-calibration"],
        schedule: {
          ...schema["x-lm-atelier-edit-calibration"].schedule,
          minimum_effective_steps: {
            localized: 2,
            replacement: 0,
            global: 3,
          },
        },
      },
    })).toBeNull();
  });
});