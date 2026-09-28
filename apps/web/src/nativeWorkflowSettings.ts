export const NATIVE_SETTINGS_KEY = "x-lm-atelier-graph-settings";
const inputNames: Record<string, string> = {
  width: "width", height: "height", seed: "seed", noise_seed: "seed", steps: "steps", cfg: "cfg",
  sampler: "sampler", sampler_name: "sampler", scheduler: "scheduler", denoise: "denoise",
  batch_size: "batch_size", frames: "frames", frame_count: "frames", fps: "fps", frame_rate: "fps",
  aspect_ratio: "aspect_ratio", megapixels: "megapixels", megapixel_budget: "megapixels",
  round_to_multiple: "round_to_multiple",
};
export const NATIVE_SETTING_KEYS = new Set(Object.values(inputNames));
export const UNMAPPED_SETTING_REASON = "This setting is not mapped to an editable input in the workflow.";
const reasons: Record<string, string> = {
  read_only: "The workflow declares this control read-only.",
  dynamic: "Changing this choice changes the node's inputs. Use the native editor.",
  linked: "Supplied by a connected node in the workflow.",
  primitive: "Fixed by a primitive node in the workflow.",
  disconnected: "This node does not contribute to an output.",
  unsupported: "This native control cannot be edited through generation settings.",
};

export interface NativeFixedSetting {
  nodeId: string;
  inputName: string;
  label: string;
  explanation: string;
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function strings(value: unknown, keys: string[]): value is Record<string, string> {
  return record(value) && (Object.keys(value).length === keys.length
    || Object.keys(value).length === keys.length + 1 && typeof value.declared_name === "string" && value.declared_name.length > 0)
    && keys.every((key) => typeof value[key] === "string" && value[key].length > 0);
}

function knownInput(item: Record<string, string>): boolean {
  const declared = item.declared_name ?? item.input_name;
  return Object.hasOwn(inputNames, declared);
}

export function nativeWorkflowSettingParameters(
  schema: Record<string, unknown> | undefined, setting: string,
): string[] {
  const raw = schema?.[NATIVE_SETTINGS_KEY];
  if (!record(raw) || nativeWorkflowSettings(schema) === null) return [];
  return (raw.bindings as Record<string, string>[])
    .filter((binding) => inputNames[binding.declared_name ?? binding.input_name] === setting)
    .map((binding) => binding.parameter);
}

export function nativeWorkflowUnboundParameters(schema: Record<string, unknown> | undefined): string[] {
  const raw = schema?.[NATIVE_SETTINGS_KEY];
  if (!record(raw) || nativeWorkflowSettings(schema) === null) return [];
  return [...(raw.unbound_parameters ?? []) as string[]];
}

export function nativeWorkflowSettings(
  schema: Record<string, unknown> | undefined,
): NativeFixedSetting[] | null {
  const raw = schema?.[NATIVE_SETTINGS_KEY];
  const properties = schema?.properties;
  if (!record(raw) || !record(properties) || raw.version !== 1 || !Array.isArray(raw.bindings)
    || Object.keys(raw).some((key) => !["version", "bindings", "fixed", "unbound_parameters"].includes(key))
    || (raw.unbound_parameters !== undefined && !Array.isArray(raw.unbound_parameters))
    || (raw.fixed !== undefined && !Array.isArray(raw.fixed))) return null;
  const paths = new Set<string>();
  const parameters = new Set<string>();
  for (const binding of raw.bindings) {
    if (!strings(binding, ["parameter", "node_id", "input_name"])
      || !knownInput(binding) || parameters.has(binding.parameter)
      || !record(properties[binding.parameter])) return null;
    const path = JSON.stringify([binding.node_id, binding.input_name]);
    if (paths.has(path)) return null;
    paths.add(path);
    parameters.add(binding.parameter);
  }
  const fixed: NativeFixedSetting[] = [];
  for (const item of (raw.fixed ?? []) as unknown[]) {
    if (!strings(item, ["node_id", "input_name", "label", "reason"])
      || !knownInput(item) || !Object.hasOwn(reasons, item.reason)) return null;
    const path = JSON.stringify([item.node_id, item.input_name]);
    if (paths.has(path)) return null;
    paths.add(path);
    fixed.push({ nodeId: item.node_id, inputName: item.input_name,
      label: item.label, explanation: reasons[item.reason] });
  }
  for (const parameter of (raw.unbound_parameters ?? []) as unknown[]) {
    if (typeof parameter !== "string" || !Object.hasOwn(inputNames, parameter)
      || parameters.has(parameter) || !record(properties[parameter])) return null;
    parameters.add(parameter);
  }
  return fixed;
}
