import { describe, expect, it } from "vitest";
import {
  normalizeSettingsForFields,
  promptPreviewSettings,
  resolveCapabilitySettings,
  resolveWorkflowSettings,
  videoLengthDelivery,
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
  it("projects a workflow's measured frame window as one seconds control", () => {
    const fpsField: SettingField = {
      ...videoField,
      key: "fps",
      label: "FPS",
      type: "number",
      default: 24,
      minimum: 1,
      maximum: 120,
    };
    const fields = resolveWorkflowSettings(
      [{ ...videoField, minimum: 1, maximum: 1024 }, fpsField],
      {
        type: "object",
        properties: {
          frames: { type: "integer", default: 49, minimum: 17, maximum: 81 },
          fps: { type: "number", const: 16 },
        },
        "x-lm-atelier-video-length": {
          version: 1,
          frames_parameter: "frames",
          fps_parameter: "fps",
          fps_numerator: 16,
          fps_denominator: 1,
          frame_alignment: 16,
          frame_offset: 1,
        },
      },
    );

    expect(fields.map((field) => field.key)).toEqual(["duration_seconds"]);
    const length = fields[0]!;
    expect(length).toMatchObject({
      label: "Length (seconds)",
      default: 49 / 16,
      minimum: 17 / 16,
      maximum: 81 / 16,
    });
    expect(videoLengthDelivery(length, 3)).toEqual({
      frames: 49,
      deliveredSeconds: 49 / 16,
    });
    expect(normalizeSettingsForFields({ duration_seconds: 3, frames: 49 }, fields)).toEqual({
      duration_seconds: 3,
    });
  });

  it("fails closed for extended or unreduced video-length contracts", () => {
    const fpsField: SettingField = { ...videoField, key: "fps", type: "number", default: 16 };
    const contract: Record<string, unknown> = {
      version: 1,
      frames_parameter: "frames",
      fps_parameter: "fps",
      fps_numerator: 32,
      fps_denominator: 2,
      frame_alignment: 16,
      frame_offset: 1,
      extra: true,
    };
    const schema = {
      type: "object",
      properties: {
        frames: { type: "integer", default: 49, minimum: 17, maximum: 81 },
        fps: { type: "number", const: 16 },
      },
      "x-lm-atelier-video-length": contract,
    };

    expect(resolveWorkflowSettings([videoField, fpsField], schema)).toHaveLength(2);
    delete contract.extra;
    expect(resolveWorkflowSettings([videoField, fpsField], schema)).toHaveLength(2);
  });

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
  it("derives bounded low-cost preview values from capability fields", () => {
    const field = (
      key: string,
      defaultValue: number,
      minimum: number | null,
      maximum: number | null,
      overrides: Partial<SettingField> = {},
    ): SettingField => ({
      ...imageField,
      key,
      label: key,
      type: "integer",
      default: defaultValue,
      minimum,
      maximum,
      ...overrides,
    });

    expect(promptPreviewSettings([
      field("width", 1024, 256, 2048, { multiple_of: 64 }),
      field("height", 384, 256, 2048, { multiple_of: 64 }),
      field("steps", 30, 12, 100),
      field("num_frames", 49, 1, 81),
      field("duration_seconds", 6, 1, 10),
      field("batch_size", 4, 1, 8),
      field("seed", -1, -1, 2 ** 31),
      field("image_width", 1024, 256, 2048, { scope: "load" }),
      field("output_height", 1024, 256, 2048, { available: false }),
    ])).toEqual({
      width: 512,
      height: 384,
      steps: 12,
      num_frames: 16,
      duration_seconds: 2,
      batch_size: 1,
    });
  });
});
