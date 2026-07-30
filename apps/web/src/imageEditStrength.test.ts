import { describe, expect, it } from "vitest";
import strengthFixtures from "../../../services/api/tests/fixtures/image_edit_strength_v1.json";
import {
  calibratedImageEditStrength,
  estimateImageEditStrength,
  resolveImageEditStrengthMode,
  workflowImageEditCalibration,
} from "./imageEditStrength";

type StrengthFixture = {
  name: string;
  prompt: string;
  scope: "minimal" | "localized" | "replacement" | "global" | "fallback";
  confidence: "low" | "medium" | "high";
  minimum: number;
  maximum: number;
};

describe("estimateImageEditStrength", () => {
  const cases = strengthFixtures as StrengthFixture[];

  it.each(cases)("$name", ({ prompt, scope, confidence, minimum, maximum }) => {
    const estimate = estimateImageEditStrength(prompt);

    expect(estimate.scope).toBe(scope);
    expect(estimate.confidence).toBe(confidence);
    expect(estimate.value).toBeGreaterThanOrEqual(minimum);
    expect(estimate.value).toBeLessThanOrEqual(maximum);
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