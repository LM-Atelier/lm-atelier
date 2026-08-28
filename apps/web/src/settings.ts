import type { EngineCapabilities, EngineRole, SettingField } from "./types";

export interface VideoLengthControl {
  frames_parameter: string;
  fps_parameter: string;
  fps_numerator: number;
  fps_denominator: number;
  frame_alignment: number;
  frame_offset: number;
  minimum_frames: number;
  maximum_frames: number;
}

type VideoLengthSettingField = SettingField & {
  video_length?: VideoLengthControl | null;
};

const VIDEO_LENGTH_CONTRACT_KEYS = new Set([
  "version",
  "frames_parameter",
  "fps_parameter",
  "fps_numerator",
  "fps_denominator",
  "frame_alignment",
  "frame_offset",
]);
const MAX_VIDEO_LENGTH_COMPONENT = 1_000_000;
const MAX_VIDEO_LENGTH_FRAMES = 2_147_483_647;

function greatestCommonDivisor(left: number, right: number): number {
  let a = Math.abs(left);
  let b = Math.abs(right);
  while (b !== 0) [a, b] = [b, a % b];
  return a;
}

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
  if (!inputSchema || !properties) return fields;

  const videoLength = workflowVideoLengthField(fields, inputSchema, properties);
  const hiddenVideoKeys = videoLength?.video_length
    ? new Set([videoLength.video_length.frames_parameter, videoLength.video_length.fps_parameter])
    : new Set<string>();

  const baseKeys = new Set(fields.map((field) => field.key));
  const resolved = fields.filter((field) => !hiddenVideoKeys.has(field.key)).map((field) => {
    const property = properties[field.key];
    return isRecord(property) ? workflowField(field.key, property, field) ?? field : field;
  });
  if (videoLength) resolved.push(videoLength);
  for (const [key, property] of Object.entries(properties)) {
    if (
      baseKeys.has(key)
      || hiddenVideoKeys.has(key)
      || !isRecord(property)
      || isReservedSettingKey(key)
      || !("default" in property || "const" in property || Array.isArray(property.enum))
    ) continue;
    const custom = workflowField(key, property);
    if (custom) resolved.push(custom);
  }
  return resolved;
}

function workflowVideoLengthField(
  fields: SettingField[],
  inputSchema: Record<string, unknown>,
  properties: Record<string, unknown>,
): VideoLengthSettingField | null {
  const raw = inputSchema["x-lm-atelier-video-length"];
  if (
    !isRecord(raw)
    || raw.version !== 1
    || Object.keys(raw).length !== VIDEO_LENGTH_CONTRACT_KEYS.size
    || Object.keys(raw).some((key) => !VIDEO_LENGTH_CONTRACT_KEYS.has(key))
  ) return null;
  const framesParameter = raw.frames_parameter;
  const fpsParameter = raw.fps_parameter;
  const numerator = raw.fps_numerator;
  const denominator = raw.fps_denominator;
  const alignment = raw.frame_alignment;
  const offset = raw.frame_offset;
  if (
    typeof framesParameter !== "string"
    || typeof fpsParameter !== "string"
    || framesParameter.length === 0
    || framesParameter.length > 200
    || fpsParameter.length === 0
    || fpsParameter.length > 200
    || framesParameter === fpsParameter
    || framesParameter === "duration_seconds"
    || fpsParameter === "duration_seconds"
    || ![numerator, denominator, alignment, offset].every(Number.isInteger)
    || Number(numerator) <= 0
    || Number(numerator) > MAX_VIDEO_LENGTH_COMPONENT
    || Number(denominator) <= 0
    || Number(denominator) > MAX_VIDEO_LENGTH_COMPONENT
    || greatestCommonDivisor(Number(numerator), Number(denominator)) !== 1
    || Number(alignment) <= 0
    || Number(alignment) > MAX_VIDEO_LENGTH_FRAMES
    || Number(offset) < 0
    || Number(offset) > MAX_VIDEO_LENGTH_FRAMES
    || Number(offset) >= Number(alignment)
  ) return null;
  const framesSchema = properties[framesParameter];
  const fpsSchema = properties[fpsParameter];
  if (!isRecord(framesSchema) || !isRecord(fpsSchema) || "duration_seconds" in properties) {
    return null;
  }
  const minimumFrames = framesSchema.minimum;
  const maximumFrames = framesSchema.maximum;
  const defaultFrames = framesSchema.default;
  const declaredFps = "const" in fpsSchema ? fpsSchema.const : fpsSchema.default;
  if (
    framesSchema.type !== "integer"
    || ![minimumFrames, maximumFrames, defaultFrames].every(Number.isInteger)
    || fpsSchema.type !== "integer" && fpsSchema.type !== "number"
    || typeof declaredFps !== "number"
    || !Number.isFinite(declaredFps)
    || Math.abs(declaredFps - Number(numerator) / Number(denominator)) > 1e-12
  ) return null;
  const minimum = Number(minimumFrames);
  const maximum = Number(maximumFrames);
  const defaultValue = Number(defaultFrames);
  const frameBase = fields.find((field) => field.key === framesParameter && field.type === "integer");
  const fpsBase = fields.find((field) => field.key === fpsParameter && ["integer", "number"].includes(field.type));
  if (
    !frameBase
    || !fpsBase
    || minimum < 0
    || maximum > MAX_VIDEO_LENGTH_FRAMES
    || minimum > defaultValue
    || defaultValue > maximum
    || [minimum, maximum, defaultValue].some((value) => (value - Number(offset)) % Number(alignment) !== 0)
    || frameBase.minimum != null && minimum < frameBase.minimum
    || frameBase.maximum != null && maximum > frameBase.maximum
    || fpsBase.minimum != null && declaredFps < fpsBase.minimum
    || fpsBase.maximum != null && declaredFps > fpsBase.maximum
  ) return null;
  const fps = Number(numerator) / Number(denominator);
  return {
    key: "duration_seconds",
    label: "Length (seconds)",
    type: "number",
    default: defaultValue / fps,
    minimum: minimum / fps,
    maximum: maximum / fps,
    step: 0.01,
    multiple_of: null,
    choices: [],
    scope: "workflow",
    visibility: "basic",
    restart_required: false,
    available: true,
    unavailable_reason: null,
    help: "Choose a length in seconds. This workflow aligns the request to a supported frame count and shows the delivered length.",
    video_length: {
      frames_parameter: framesParameter,
      fps_parameter: fpsParameter,
      fps_numerator: Number(numerator),
      fps_denominator: Number(denominator),
      frame_alignment: Number(alignment),
      frame_offset: Number(offset),
      minimum_frames: minimum,
      maximum_frames: maximum,
    },
  };
}

export function videoLengthDelivery(
  field: SettingField,
  requestedSeconds: number,
): { frames: number; deliveredSeconds: number } | null {
  const control = (field as VideoLengthSettingField).video_length;
  if (!control || !Number.isFinite(requestedSeconds)) return null;
  const fps = control.fps_numerator / control.fps_denominator;
  const target = requestedSeconds * fps;
  const lower = control.frame_offset
    + Math.floor((target - control.frame_offset) / control.frame_alignment)
      * control.frame_alignment;
  const upper = target === lower ? lower : lower + control.frame_alignment;
  const candidates = [lower, upper].filter(
    (value) => value >= control.minimum_frames && value <= control.maximum_frames,
  );
  const frames = candidates.length
    ? candidates.sort((left, right) => Math.abs(left - target) - Math.abs(right - target) || left - right)[0]!
    : target < control.minimum_frames
      ? control.minimum_frames
      : control.maximum_frames;
  return { frames, deliveredSeconds: frames / fps };
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
