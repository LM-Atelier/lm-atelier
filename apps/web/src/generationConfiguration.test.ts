import { describe, expect, it } from "vitest";
import { recordedGenerationDetails } from "./generationConfiguration";

describe("recorded generation configuration", () => {
  it("keeps captured names, zero weights and seed without exposing raw metadata", () => {
    const record = recordedGenerationDetails({
      model: { profile_name: "Original model", path: "private model path" },
      workflow: { family_name: "Original workflow", version: 2 },
      resolved_settings: { seed: 0, cfg: 0, width: 512, prompt: "neutral hidden prompt", custom_path: "private setting path" },
      auxiliary_assets: { lora_stack: [
        { name: "Watercolor", enabled: true, model_strength: 0, clip_strength: 0.6, trigger_words: ["hidden trigger"] },
        { name: "Soft light", enabled: false, model_strength: 0.4, clip_strength: 0 },
      ] },
    });
    expect(record.identity?.model_profile_name).toBe("Original model");
    expect(record.settings).toContainEqual({ label: "Seed", value: "0" });
    expect(record.addedLoras).toEqual([
      { name: "Watercolor", enabled: true, modelStrength: 0, clipStrength: 0.6 },
      { name: "Soft light", enabled: false, modelStrength: 0.4, clipStrength: 0 },
    ]);
    expect(JSON.stringify(record)).not.toMatch(/private|hidden/);
  });

  it("distinguishes missing records from an explicitly empty added stack", () => {
    expect(recordedGenerationDetails(null).addedLorasRecorded).toBe(false);
    expect(recordedGenerationDetails({ resolved_settings: { width: 512 } }).addedLorasRecorded).toBe(false);
    expect(recordedGenerationDetails({ resolved_settings: { loras: [] } }).addedLorasRecorded).toBe(true);
    expect(recordedGenerationDetails({ resolved_settings: { loras: [{ asset_id: "opaque" }] } }).addedLorasRecorded).toBe(false);
  });

  it("shows native override weights without inventing uncaptured names or unchanged fields", () => {
    const record = recordedGenerationDetails({ workflow_lora: { composition: {
      added_provenance: [],
      workflow_native: { graph_resolution: { overrides: [{
        slot_id: "opaque slot", asset_sha256: "opaque digest",
        changes: [{ field: "model_strength", effective_value: 0.25, authored_value: 1 }],
      }] } },
    } } });
    expect(record.workflowLoras).toEqual([{
      name: "Workflow LoRA 1 (name not recorded)", modelStrength: 0.25, clipStrength: null, enabled: null,
    }]);
    expect(JSON.stringify(record)).not.toContain("opaque");
  });

  it("refuses malformed scalar values instead of coercing them into configuration", () => {
    const record = recordedGenerationDetails({
      resolved_settings: { seed: null, cfg: Infinity, steps: {}, fps: false },
      auxiliary_assets: { lora_stack: [null, { name: 12, enabled: "yes", model_strength: "1", clip_strength: NaN }] },
    });
    expect(record.settings).toEqual([]);
    expect(record.addedLoras).toEqual([{
      name: "LoRA name not recorded", enabled: null, modelStrength: null, clipStrength: null,
    }]);
  });
});
