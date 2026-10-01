import { generationIdentityFromProvenance } from "./generationIdentity";
import type { GenerationIdentity } from "./types";

export interface RecordedLora {
  name: string;
  modelStrength: number | null;
  clipStrength: number | null;
  enabled: boolean | null;
}

export interface RecordedGenerationDetails {
  identity: GenerationIdentity | null;
  settings: { label: string; value: string }[];
  addedLoras: RecordedLora[];
  addedLorasRecorded: boolean;
  workflowLoras: RecordedLora[];
}

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() && value.length <= 500
    ? value.trim() : null;
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

const SETTING_LABELS: Record<string, string> = {
  seed: "Seed", width: "Width", height: "Height", steps: "Steps", cfg: "Guidance",
  sampler: "Sampler", scheduler: "Scheduler", denoise: "Denoise",
  batch_size: "Batch size", frames: "Frames", fps: "Frame rate", codec: "Codec",
  duration_seconds: "Duration (seconds)", upscale_factor: "Enlargement factor",
};

/** Project only recorded display fields; prompts, paths and resolver receipts
 * never become an arbitrary metadata dump. No current settings are consulted. */
export function recordedGenerationDetails(value: unknown): RecordedGenerationDetails {
  const provenance = object(value);
  const settings = object(provenance?.resolved_settings);
  const auxiliary = object(provenance?.auxiliary_assets);
  const composition = object(object(provenance?.workflow_lora)?.composition);
  const stack = composition?.added_provenance ?? auxiliary?.lora_stack;
  const addedLoras = Array.isArray(stack) ? stack.flatMap((value) => {
    const item = object(value);
    if (!item) return [];
    return [{
      name: text(item.name) ?? "LoRA name not recorded",
      modelStrength: number(item.model_strength),
      clipStrength: number(item.clip_strength),
      enabled: typeof item.enabled === "boolean" ? item.enabled : null,
    }];
  }) : [];
  const native = object(composition?.workflow_native);
  const overrides = object(native?.graph_resolution)?.overrides;
  const workflowLoras = Array.isArray(overrides) ? overrides.flatMap((value, index) => {
    const item = object(value);
    if (!item || !Array.isArray(item.changes)) return [];
    const fields = new Map(item.changes.flatMap((value) => {
      const change = object(value);
      return change && typeof change.field === "string"
        ? [[change.field, change.effective_value] as const] : [];
    }));
    return [{
      name: `Workflow LoRA ${index + 1} (name not recorded)`,
      modelStrength: number(fields.get("model_strength")),
      clipStrength: number(fields.get("clip_strength")),
      enabled: typeof fields.get("enabled") === "boolean"
        ? fields.get("enabled") as boolean : null,
    }];
  }) : [];
  return {
    identity: generationIdentityFromProvenance(provenance),
    settings: Object.entries(SETTING_LABELS).flatMap(([key, label]) => {
      const value = settings?.[key];
      const display = number(value) !== null ? String(value) : text(value);
      return display !== null ? [{ label, value: display }] : [];
    }),
    addedLoras,
    addedLorasRecorded: Array.isArray(stack)
      || Array.isArray(settings?.loras) && settings.loras.length === 0,
    workflowLoras,
  };
}
