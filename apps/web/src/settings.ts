import type { EngineCapabilities, EngineRole, SettingField } from "./types";

export function resolveCapabilitySettings(
  engine: EngineCapabilities | undefined,
  role: EngineRole,
): SettingField[] {
  return engine?.settings_by_role?.[role] ?? engine?.settings ?? [];
}

export function resolveWorkflowSettings(
  fields: SettingField[],
  inputSchema: Record<string, unknown> | undefined,
): SettingField[] {
  const properties = isRecord(inputSchema?.properties) ? inputSchema.properties : null;
  if (!properties) return fields;

  const baseKeys = new Set(fields.map((field) => field.key));
  const resolved = fields.map((field) => {
    const property = properties[field.key];
    return isRecord(property) ? workflowField(field.key, property, field) ?? field : field;
  });
  for (const [key, property] of Object.entries(properties)) {
    if (
      baseKeys.has(key)
      || !isRecord(property)
      || isReservedSettingKey(key)
      || !("default" in property || "const" in property || Array.isArray(property.enum))
    ) continue;
    const custom = workflowField(key, property);
    if (custom) resolved.push(custom);
  }
  return resolved;
}

export function normalizeSettingsForFields(
  values: Record<string, unknown>,
  fields: SettingField[],
): Record<string, unknown> {
  const definitions = new Map(fields.map((field) => [field.key, field]));
  return Object.fromEntries(Object.entries(values).filter(([key, value]) => {
    const field = definitions.get(key);
    if (!field) return false;
    if (field.choices.length > 0 && !field.choices.includes(value)) return false;
    if (typeof value === "number") {
      if (field.minimum != null && value < field.minimum) return false;
      if (field.maximum != null && value > field.maximum) return false;
      if (field.multiple_of != null) {
        const quotient = value / field.multiple_of;
        if (Math.abs(quotient - Math.round(quotient)) > 1e-9) return false;
      }
    }
    return true;
  }));
}

function workflowField(
  key: string,
  schema: Record<string, unknown>,
  base?: SettingField,
): SettingField | null {
  const supportedTypes = new Set<SettingField["type"]>([
    "boolean", "integer", "number", "string", "enum", "array", "object",
  ]);
  const declared = typeof schema.type === "string" && supportedTypes.has(schema.type as SettingField["type"])
    ? schema.type as SettingField["type"]
    : null;
  if (
    base
    && declared
    && declared !== base.type
    && !(base.type === "enum" && ["boolean", "integer", "number", "string"].includes(declared))
  ) return null;
  const inferred = base?.type ?? declared ?? inferType(schema.const ?? schema.default);
  if (!inferred) return null;
  const declaredChoices = "const" in schema
    ? [schema.const]
    : Array.isArray(schema.enum)
      ? schema.enum
      : base?.choices ?? [];
  if (
    base?.choices.length
    && declaredChoices.some((choice) => !base.choices.includes(choice))
  ) return null;
  const choices = declaredChoices.length ? declaredChoices : base?.choices ?? [];
  const defaultValue = "const" in schema
    ? schema.const
    : "default" in schema
      ? schema.default
      : base?.default ?? choices[0] ?? null;
  const declaredMinimum = typeof schema.minimum === "number" ? schema.minimum : null;
  let minimum = declaredMinimum == null
    ? base?.minimum ?? null
    : Math.max(declaredMinimum, base?.minimum ?? declaredMinimum);
  if (key === "seed" && base?.default === -1 && defaultValue === -1) {
    minimum = Math.min(minimum ?? -1, -1);
  }
  const declaredMaximum = typeof schema.maximum === "number" ? schema.maximum : null;
  const maximum = declaredMaximum == null
    ? base?.maximum ?? null
    : Math.min(declaredMaximum, base?.maximum ?? declaredMaximum);
  if (minimum != null && maximum != null && minimum > maximum) return null;
  const declaredMultiple = typeof schema.multipleOf === "number" ? schema.multipleOf : null;
  if (
    base?.multiple_of != null
    && declaredMultiple != null
    && !isStricterMultiple(declaredMultiple, base.multiple_of)
  ) return null;
  const fixedHelp = "const" in schema ? `Fixed by this workflow at ${String(schema.const)}.` : "";
  const declaredVisibility = schema["x-lm-atelier-visibility"];
  const visibility = (
    declaredVisibility === "basic"
    || declaredVisibility === "advanced"
    || declaredVisibility === "expert"
  )
    ? declaredVisibility
    : base?.visibility ?? "advanced";
  return {
    key,
    label: typeof schema.title === "string" ? schema.title : base?.label ?? titleCase(key),
    type: inferred,
    default: defaultValue,
    minimum,
    maximum,
    step: typeof schema.multipleOf === "number" ? schema.multipleOf : base?.step ?? null,
    multiple_of: declaredMultiple ?? base?.multiple_of ?? null,
    choices,
    scope: base?.scope ?? "workflow",
    visibility,
    restart_required: base?.restart_required ?? false,
    available: base?.available ?? true,
    unavailable_reason: base?.unavailable_reason ?? null,
    help: [typeof schema.description === "string" ? schema.description : base?.help ?? "", fixedHelp]
      .filter(Boolean)
      .join(" "),
  };
}

function inferType(value: unknown): SettingField["type"] | null {
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "number") return Number.isInteger(value) ? "integer" : "number";
  if (typeof value === "string") return "string";
  if (Array.isArray(value)) return "array";
  if (isRecord(value)) return "object";
  return null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isReservedSettingKey(key: string): boolean {
  const normalized = key.toLowerCase();
  return normalized.startsWith("_")
    || [
      "input_image",
      "input_images",
      "messages",
      "operation",
      "parameters",
      "prompt",
      "run_id",
      "tools",
      "workflow",
    ].includes(normalized)
    || /^input_image_[0-9]+$/i.test(normalized);
}

function isStricterMultiple(candidate: number, base: number): boolean {
  const quotient = candidate / base;
  return candidate >= base && Math.abs(quotient - Math.round(quotient)) <= 1e-9;
}

function titleCase(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}
const PROMPT_PREVIEW_LIMITS: ReadonlyArray<[readonly string[], number]> = [
  [["steps", "num_inference_steps", "inference_steps", "sampling_steps"], 8],
  [["width", "image_width", "output_width"], 512],
  [["height", "image_height", "output_height"], 512],
  [["frames", "num_frames", "frame_count"], 16],
  [["duration", "duration_seconds", "video_duration"], 2],
  [["batch_size", "batch_count", "num_images", "image_count"], 1],
];

export function promptPreviewSettings(fields: SettingField[]): Record<string, unknown> {
  const settings: Record<string, unknown> = {};
  for (const field of fields) {
    if (!field.available || field.scope === "load" || !["integer", "number"].includes(field.type)) {
      continue;
    }
    const normalizedKey = field.key.trim().toLowerCase().replaceAll("-", "_");
    const target = PROMPT_PREVIEW_LIMITS.find(([keys]) => keys.includes(normalizedKey))?.[1];
    if (target === undefined) continue;
    const baseline = typeof field.default === "number" && Number.isFinite(field.default)
      ? Math.min(field.default, target)
      : target;
    const minimum = field.minimum ?? Number.NEGATIVE_INFINITY;
    const maximum = field.maximum ?? Number.POSITIVE_INFINITY;
    let value = Math.min(Math.max(baseline, minimum), maximum);
    const multiple = field.multiple_of ?? field.step;
    if (multiple != null && multiple > 0 && Number.isFinite(multiple)) {
      const origin = Number.isFinite(minimum) ? minimum : 0;
      value = origin + Math.floor((value - origin) / multiple) * multiple;
      value = Math.min(Math.max(value, minimum), maximum);
    }
    settings[field.key] = field.type === "integer" ? Math.round(value) : value;
  }
  return settings;
}


/** How much of a workflow's settings a person has asked to see.
 *
 * Lived in App.tsx while two different panels filtered by it, which meant the
 * shell owned a fact about settings. It belongs here, with the settings whose
 * visibility it decides.
 */
export type Visibility = "basic" | "advanced" | "expert";

export const visibilityRank: Record<Visibility, number> = {
  basic: 0,
  advanced: 1,
  expert: 2,
};
