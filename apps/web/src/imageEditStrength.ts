export const IMAGE_EDIT_STRENGTH_MODE_KEY = "_image_edit_strength_mode";
export const IMAGE_EDIT_CALIBRATION_SCHEMA_KEY = "x-lm-atelier-edit-calibration";

export type ImageEditScope = "minimal" | "localized" | "replacement" | "global" | "fallback";
export type ImageEditStrengthMode = "auto" | "manual";

export interface ImageEditStrengthEstimate {
  scope: ImageEditScope;
  value: number;
  confidence: "low" | "medium" | "high";
}
export interface WorkflowImageEditCalibration {
  parameter: string;
  minimum: number;
  maximum: number;
  recommended: Record<ImageEditScope, number>;
  stepsParameter: string | null;
  minimumEffectiveSteps: Partial<Record<ImageEditScope, number>>;
}

const strength: Record<ImageEditScope, number> = {
  minimal: 0.38,
  localized: 0.50,
  replacement: 0.66,
  global: 0.82,
  fallback: 0.56,
};
const globalPhrases = [
  "change the entire image", "change the whole image", "complete transformation",
  "different composition", "new composition", "new scene", "oil painting",
  "watercolor painting",
];
const globalWords = new Set(["recompose", "restyle", "stylize", "transform", "watercolor"]);
const replacementPhrases = [
  "change the background", "different background", "new background", "new clothes",
  "new clothing", "new hairstyle", "new outfit", "replace the background",
];
const replacementTargets = new Set([
  "background", "clothes", "clothing", "coat", "dress", "hair", "hairstyle",
  "jacket", "object", "outfit", "shirt", "suit", "wardrobe",
]);
const replacementVerbs = new Set(["change", "dress", "give", "make", "replace", "swap"]);
const localizedPhrases = [
  "add a", "add an", "make it blue", "make it green", "make it red", "remove the",
];
const localizedWords = new Set(["add", "erase", "insert", "recolor", "remove"]);
const minimalPhrases = [
  "color correction", "colour correction", "make it brighter", "make it darker",
  "slightly brighter", "slightly darker", "subtle change", "warm lighting",
];
const minimalWords = new Set([
  "brightness", "contrast", "exposure", "lighting", "sharpen", "slight", "slightly",
  "subtle",
]);
const preservationPhrases = [
  "do not alter", "do not change", "don t alter", "don t change", "keep everything else",
  "preserve identity", "preserve the rest", "without altering", "without changing",
];

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function workflowImageEditCalibration(
  schema: Record<string, unknown> | undefined,
): WorkflowImageEditCalibration | null {
  const contract = record(schema?.[IMAGE_EDIT_CALIBRATION_SCHEMA_KEY]);
  const editStrength = record(contract?.edit_strength);
  const recommended = record(editStrength?.recommended);
  const properties = record(schema?.properties);
  const parameter = editStrength?.parameter;
  const minimum = finiteNumber(editStrength?.minimum);
  const maximum = finiteNumber(editStrength?.maximum);
  const parameterSchema = typeof parameter === "string"
    ? record(properties?.[parameter])
    : null;
  if (
    contract?.version !== 1
    || typeof parameter !== "string"
    || !parameter
    || minimum === null
    || maximum === null
    || minimum >= maximum
    || parameterSchema?.type !== "number"
    || !(Object.hasOwn(parameterSchema, "default") || Object.hasOwn(parameterSchema, "const"))
    || !recommended
  ) return null;
  const requiredScopes: ImageEditScope[] = ["minimal", "localized", "replacement", "global"];
  const normalizedRecommended = {} as Record<ImageEditScope, number>;
  for (const scope of requiredScopes) {
    const value = finiteNumber(recommended[scope]);
    if (value === null || value < minimum || value > maximum) return null;
    normalizedRecommended[scope] = value;
  }
  const fallback = finiteNumber(recommended.fallback);
  if (fallback !== null && (fallback < minimum || fallback > maximum)) return null;
  normalizedRecommended.fallback = fallback
    ?? Math.min(Math.max(strength.fallback, minimum), maximum);
  const rawSchedule = contract.schedule;
  const schedule = record(rawSchedule);
  if (rawSchedule !== undefined && !schedule) return null;
  let stepsParameter: string | null = null;
  const minimumEffectiveSteps: Partial<Record<ImageEditScope, number>> = {};
  if (schedule) {
    const candidate = schedule.steps_parameter;
    const stepsSchema = typeof candidate === "string" ? record(properties?.[candidate]) : null;
    if (
      typeof candidate !== "string"
      || !stepsSchema
      || !["integer", "number"].includes(String(stepsSchema.type))
      || !(Object.hasOwn(stepsSchema, "default") || Object.hasOwn(stepsSchema, "const"))
    ) return null;
    stepsParameter = candidate;
    const rawMinimumSteps = record(schedule.minimum_effective_steps);
    if (!rawMinimumSteps) return null;
    for (const scope of ["localized", "replacement", "global"] as ImageEditScope[]) {
      const value = finiteNumber(rawMinimumSteps[scope]);
      if (value === null || value <= 0 || value > 10_000) return null;
      minimumEffectiveSteps[scope] = value;
    }
  }
  return {
    parameter,
    minimum,
    maximum,
    recommended: normalizedRecommended,
    stepsParameter,
    minimumEffectiveSteps,
  };
}
function normalize(prompt: string): { text: string; words: Set<string> } {
  let normalized = "";
  for (const character of prompt.toLocaleLowerCase("en-US")) {
    normalized += /[a-z0-9]/.test(character) ? character : " ";
  }
  const text = normalized.trim().split(/\s+/).filter(Boolean).join(" ");
  return { text, words: new Set(text.split(" ").filter(Boolean)) };
}

function hasPhrase(text: string, phrases: string[]): boolean {
  const padded = ` ${text} `;
  return phrases.some((phrase) => padded.includes(` ${phrase} `));
}

function intersects(words: Set<string>, candidates: Set<string>): boolean {
  return [...candidates].some((candidate) => words.has(candidate));
}

export function estimateImageEditStrength(
  prompt: string,
  minimum = 0,
  maximum = 1,
  recommended: Record<ImageEditScope, number> = strength,
): ImageEditStrengthEstimate {
  const { text, words } = normalize(prompt);
  const minimal = hasPhrase(text, minimalPhrases) || intersects(words, minimalWords);
  const preservation = hasPhrase(text, preservationPhrases);
  let scope: ImageEditScope;
  let confidence: ImageEditStrengthEstimate["confidence"];
  if (hasPhrase(text, globalPhrases) || intersects(words, globalWords)) {
    scope = "global";
    confidence = "high";
  } else if (preservation && minimal) {
    scope = "minimal";
    confidence = "high";
  } else if (
    hasPhrase(text, replacementPhrases)
    || (intersects(words, replacementTargets) && intersects(words, replacementVerbs))
  ) {
    scope = "replacement";
    confidence = "high";
  } else if (minimal) {
    scope = "minimal";
    confidence = "high";
  } else if (hasPhrase(text, localizedPhrases) || intersects(words, localizedWords)) {
    scope = "localized";
    confidence = "medium";
  } else {
    scope = "fallback";
    confidence = "low";
  }
  return {
    scope,
    confidence,
    value: Math.round(Math.min(Math.max(recommended[scope], minimum), maximum) * 10_000) / 10_000,
  };
}
export function calibratedImageEditStrength(
  prompt: string,
  calibration: WorkflowImageEditCalibration | null,
  resolvedSteps?: unknown,
): ImageEditStrengthEstimate {
  if (!calibration) return estimateImageEditStrength(prompt);
  const estimate = estimateImageEditStrength(
    prompt,
    calibration.minimum,
    calibration.maximum,
    calibration.recommended,
  );
  const minimumSteps = calibration.minimumEffectiveSteps[estimate.scope];
  if (
    typeof resolvedSteps !== "number"
    || !Number.isFinite(resolvedSteps)
    || resolvedSteps <= 0
    || minimumSteps === undefined
  ) return estimate;
  const value = Math.min(
    Math.max(estimate.value, minimumSteps / resolvedSteps, calibration.minimum),
    calibration.maximum,
  );
  return { ...estimate, value: Math.round(value * 10_000) / 10_000 };
}