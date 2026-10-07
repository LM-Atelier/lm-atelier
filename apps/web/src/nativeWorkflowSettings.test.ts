import { describe, expect, it } from "vitest";
import { normalizeSettingsForFields, resolveWorkflowSettings } from "./settings";
import type { SettingField } from "./types";
import { nativeWorkflowSettings, NATIVE_SETTINGS_KEY } from "./nativeWorkflowSettings";

const field: SettingField = {
  key: "cfg",
  label: "CFG",
  type: "number",
  default: 7,
  minimum: 0,
  maximum: 30,
  step: 0.1,
  multiple_of: null,
  choices: [],
  scope: "workflow",
  visibility: "basic",
  restart_required: false,
  available: true,
  unavailable_reason: null,
  help: "",
};

describe("native workflow controls", () => {
  it("keeps added LoRAs alongside mapped native controls", () => {
    const fields = resolveWorkflowSettings([field], { properties: {
      cfg: { type: "number", default: 8.25, minimum: 0, maximum: 30, "x-lm-atelier-step": 0.5 },
    }, [NATIVE_SETTINGS_KEY]: { version: 1,
      bindings: [{ node_id: "1", input_name: "cfg", parameter: "cfg" }], fixed: [],
    } }, true);
    const loras = [{ asset_id: "watercolor", model_strength: 0.7, clip_strength: 0.7, enabled: true }];
    expect(fields.find((item) => item.key === "cfg")).toMatchObject({ default: 8.25, step: 0.5 });
    expect(fields.find((item) => item.key === "loras")).toMatchObject({ type: "array", available: true });
    expect(normalizeSettingsForFields({ cfg: 8.25, loras }, fields)).toEqual({ cfg: 8.25, loras });
  });

  it("explains why a declared read-only control cannot be changed", () => {
    expect(nativeWorkflowSettings({ properties: { seed: { type: "integer", readOnly: true } },
      [NATIVE_SETTINGS_KEY]: { version: 1, bindings: [], fixed: [
        { node_id: "1", input_name: "seed", label: "Seed (Source)", reason: "read_only" },
      ] },
    })).toEqual([{ nodeId: "1", inputName: "seed", label: "Seed (Source)",
      explanation: "The workflow declares this control read-only." }]);
  });

  it("rejects an availability record that contradicts a generated binding", () => {
    expect(nativeWorkflowSettings({ properties: { seed: { type: "integer", default: 42 } },
      [NATIVE_SETTINGS_KEY]: { version: 1,
        bindings: [{ node_id: "1", input_name: "seed", parameter: "seed" }],
        unbound_parameters: ["seed"],
      },
    })).toBeNull();
  });

  it("explains a named grown socket using its recorded template declaration", () => {
    expect(nativeWorkflowSettings({ properties: {}, [NATIVE_SETTINGS_KEY]: {
      version: 1, bindings: [], fixed: [{ node_id: "1", input_name: "seeds.first",
        declared_name: "seed", label: "Seed (Source)", reason: "linked" }],
    } })).toEqual([{ nodeId: "1", inputName: "seeds.first", label: "Seed (Source)",
      explanation: "Supplied by a connected node in the workflow." }]);
  });

  it("does not infer a setting from a grown socket's label", () => {
    expect(nativeWorkflowSettings({ properties: {}, [NATIVE_SETTINGS_KEY]: {
      version: 1, bindings: [], fixed: [{ node_id: "1", input_name: "seeds.seed",
        label: "Seed (Source)", reason: "linked" }],
    } })).toBeNull();
  });

  it("uses the native widget increment without rounding a valid saved value", () => {
    const fields = resolveWorkflowSettings([field], { properties: {
      cfg: { type: "number", default: 8.25, minimum: 0, maximum: 30, "x-lm-atelier-step": 0.5 },
    } });
    expect(fields[0]).toMatchObject({ default: 8.25, step: 0.5, multiple_of: null });
    expect(normalizeSettingsForFields({ cfg: 8.25 }, fields)).toEqual({ cfg: 8.25 });
  });

  it("preserves an integer-only native frame rate", () => {
    const fields = resolveWorkflowSettings([{ ...field, key: "fps", maximum: 120 }], { properties: {
      fps: { type: "integer", default: 24, minimum: 1, maximum: 120, "x-lm-atelier-step": 1 },
    } });
    expect(fields[0]).toMatchObject({ type: "integer", default: 24, step: 1 });
    expect(normalizeSettingsForFields({ fps: 23.5 }, fields)).toEqual({});
    expect(normalizeSettingsForFields({ fps: 24 }, fields)).toEqual({ fps: 24 });
  });
});
