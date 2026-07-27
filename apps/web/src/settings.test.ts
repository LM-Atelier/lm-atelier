import { describe, expect, it } from "vitest";
import {
  normalizeSettingsForFields,
  resolveCapabilitySettings,
  resolveWorkflowSettings,
} from "./settings";
import type { EngineCapabilities, SettingField } from "./types";

const imageField: SettingField = {
  key: "negative_prompt",
  label: "Negative prompt",
  type: "string",
  default: "",
  minimum: null,
  maximum: null,
  step: null,
  choices: [],
  scope: "workflow",
  visibility: "basic",
  restart_required: false,
  available: true,
  unavailable_reason: null,
  help: "",
};

const videoField: SettingField = { ...imageField, key: "frames", label: "Frames", type: "integer", default: 49 };

function mediaCapabilities(settingsByRole?: EngineCapabilities["settings_by_role"]): EngineCapabilities {
  return {
    engine: "media",
    version: "1",
    roles: ["image", "video"],
    operations: ["text_to_image", "text_to_video"],
    formats: ["mock"],
    devices: ["cpu:0"],
    streaming: false,
    tool_calling: false,
    settings: [imageField, videoField],
    settings_by_role: settingsByRole,
    healthy: true,
    details: {},
  };
}

describe("resolveCapabilitySettings", () => {
  it("selects only settings for the requested role", () => {
    const engine = mediaCapabilities({ image: [imageField], video: [videoField] });

    expect(resolveCapabilitySettings(engine, "image")).toEqual([imageField]);
    expect(resolveCapabilitySettings(engine, "video")).toEqual([videoField]);
  });

  it("falls back to legacy flat settings when the role mapping is absent", () => {
    const engine = mediaCapabilities();

    expect(resolveCapabilitySettings(engine, "image")).toEqual([imageField, videoField]);
  });
});

describe("resolveWorkflowSettings", () => {
  it("overlays workflow defaults and fixed values without dropping legacy controls", () => {
    const fields = resolveWorkflowSettings(
      [imageField, videoField],
      {
        type: "object",
        properties: {
          frames: { type: "integer", const: 81 },
          camera_strength: {
            type: "number",
            title: "Camera strength",
            default: 0.5,
            minimum: 0,
            maximum: 1,
          },
          input_image: { type: "string" },
        },
      },
    );

    expect(fields.find((field) => field.key === "negative_prompt")).toEqual(imageField);
    expect(fields.find((field) => field.key === "frames")).toMatchObject({
      default: 81,
      choices: [81],
    });
    expect(fields.find((field) => field.key === "camera_strength")).toMatchObject({
      label: "Camera strength",
      type: "number",
      default: 0.5,
      minimum: 0,
      maximum: 1,
    });
    expect(fields.some((field) => field.key === "input_image")).toBe(false);
  });

  it("removes stale overrides that conflict with the selected workflow", () => {
    const fields = resolveWorkflowSettings(
      [videoField],
      { properties: { frames: { type: "integer", const: 81 } } },
    );

    expect(normalizeSettingsForFields({ frames: 49, imaginary: true }, fields)).toEqual({});
    expect(normalizeSettingsForFields({ frames: 81 }, fields)).toEqual({ frames: 81 });
  });

  it("does not let workflow metadata weaken or replace engine controls", () => {
    const boundedFrames = {
      ...videoField,
      minimum: 1,
      maximum: 1024,
    };
    const fields = resolveWorkflowSettings(
      [boundedFrames],
      {
        properties: {
          frames: {
            type: "integer",
            default: 81,
            minimum: 0,
            maximum: 4096,
          },
        },
      },
    );

    expect(fields[0]).toMatchObject({
      type: "integer",
      minimum: 1,
      maximum: 1024,
    });
    expect(resolveWorkflowSettings(
      [boundedFrames],
      { properties: { frames: { type: "string", default: "many" } } },
    )[0]).toEqual(boundedFrames);
  });

  it("does not expose runtime-reserved workflow bindings as user settings", () => {
    const fields = resolveWorkflowSettings(
      [imageField],
      {
        properties: {
          prompt: { type: "string", default: "replacement" },
          input_image_0: { type: "string", default: "replacement.png" },
        },
      },
    );

    expect(fields).toEqual([imageField]);
  });

  it("promotes workflow controls to the requested detail level", () => {
    const fields = resolveWorkflowSettings(
      [],
      {
        properties: {
          denoise: {
            type: "number",
            title: "Edit strength",
            default: 0.9,
            minimum: 0,
            maximum: 1,
            "x-lm-atelier-visibility": "basic",
          },
        },
      },
    );

    expect(fields).toEqual([
      expect.objectContaining({
        key: "denoise",
        label: "Edit strength",
        default: 0.9,
        visibility: "basic",
      }),
    ]);
  });
});
