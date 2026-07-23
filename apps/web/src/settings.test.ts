import { describe, expect, it } from "vitest";
import { resolveCapabilitySettings } from "./settings";
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
