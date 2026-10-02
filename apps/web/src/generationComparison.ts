import type {
  ExperimentArm,
  ExperimentRefusal,
  ExperimentTrial,
  GenerationExperiment,
  GenerationExperimentRecipeDraft,
  GenerationExperimentRequest,
  ResourceEvidence,
  SeedPolicyKind,
  TrialWorkStatus,
} from "./generationExperimentTypes";
import type { OutputRatioPresetId, Run } from "./types";
import { isRecipeSettingValue } from "./workflowRecipeFields";
import type { RecipeSettingValue, WorkflowUseCasePresetCreate } from "./workflowUseCaseTypes";

export const MAX_SEED = 2_147_483_647;
const MAX_DIMENSION = 1_000_000;

export interface ComparisonChoiceDraft {
  label: string;
  profileId: string;
  revisionId: string;
}

export type ComparisonSizeDraft =
  | { mode: "preset"; presetId: OutputRatioPresetId | "" }
  | { mode: "size"; width: string; height: string };

export interface ComparisonDraft {
  prompt: string;
  negativePrompt: string;
  size: ComparisonSizeDraft;
  seed: { kind: SeedPolicyKind; number: string };
  choices: [ComparisonChoiceDraft, ComparisonChoiceDraft];
}

export const EMPTY_COMPARISON: ComparisonDraft = {
  prompt: "",
  negativePrompt: "",
  size: { mode: "size", width: "1024", height: "1024" },
  seed: { kind: "same_recorded_number", number: "" },
  choices: [
    { label: "Choice A", profileId: "", revisionId: "" },
    { label: "Choice B", profileId: "", revisionId: "" },
  ],
};

export const SEED_POLICY_LABELS: Record<SeedPolicyKind, string> = {
  same_recorded_number: "Same number for both",
  fixed_numeric: "Same starting noise (one model family only)",
  independent_deterministic: "A different number for each, from one base",
  random_per_trial: "A random number for each",
};

// Python folds case more fully than toLowerCase does: "ß" folds to "ss".
function folded(value: string): string {
  return value.toUpperCase().toLowerCase();
}

function wholeNumber(value: string, minimum: number, maximum: number): number | null {
  const trimmed = value.trim();
  if (!/^\d+$/.test(trimmed)) return null;
  const parsed = Number(trimmed);
  return Number.isSafeInteger(parsed) && parsed >= minimum && parsed <= maximum ? parsed : null;
}

/** The request a draft describes, or the reasons it cannot be sent as written. */
export function comparisonRequest(draft: ComparisonDraft, sharedPresets: OutputRatioPresetId[]):
  { request: GenerationExperimentRequest; problems: [] } | { request: null; problems: string[] } {
  const problems: string[] = [];
  const labels = draft.choices.map((choice) => choice.label.trim());
  if (labels.some((label) => label.length < 1 || label.length > 80)) problems.push("Give each choice a label of 1 to 80 characters.");
  else if (folded(labels[0]) === folded(labels[1])) problems.push("Give each choice its own label.");
  if (draft.choices.some((choice) => !choice.profileId || choice.profileId.length > 40)) problems.push("Choose an image model for each choice.");
  if (draft.choices.some((choice) => !choice.revisionId || choice.revisionId.length > 40)) problems.push("Choose a workflow for each choice.");
  if (!draft.prompt.trim()) problems.push("Write the prompt both choices share.");
  if (draft.prompt.length > 200_000) problems.push("The prompt is too long.");
  if (draft.negativePrompt.length > 20_000) problems.push("The negative prompt is too long.");
  let geometry: GenerationExperimentRequest["geometry"] | null = null;
  if (draft.size.mode === "preset") {
    if (!draft.size.presetId) problems.push("Choose a shape, or an exact size.");
    else if (!sharedPresets.includes(draft.size.presetId)) problems.push("Choose a shape both workflows can make.");
    else geometry = { mode: "preset", preset_id: draft.size.presetId };
  } else {
    const width = wholeNumber(draft.size.width, 1, MAX_DIMENSION);
    const height = wholeNumber(draft.size.height, 1, MAX_DIMENSION);
    if (width === null || height === null) problems.push("Enter a whole-number width and height.");
    else geometry = { mode: "size", width, height };
  }
  let seed: number | null = null;
  if (draft.seed.kind !== "random_per_trial" && draft.seed.number.trim()) {
    seed = wholeNumber(draft.seed.number, 0, MAX_SEED);
    if (seed === null) problems.push(`Enter a seed from 0 to ${MAX_SEED}.`);
  }
  if (draft.seed.kind === "fixed_numeric" && !draft.seed.number.trim()) problems.push("Enter the seed both choices start from.");
  if (problems.length || geometry === null) return { request: null, problems };
  return {
    request: {
      name: `${labels[0]} and ${labels[1]}`.slice(0, 200),
      operation: "text_to_image",
      prompt: draft.prompt,
      negative_prompt: draft.negativePrompt,
      geometry,
      seed_policy: { kind: draft.seed.kind, seed: draft.seed.kind === "random_per_trial" ? null : seed },
      arms: draft.choices.map((choice, index) => ({
        label: labels[index], profile_id: choice.profileId, workflow_revision_id: choice.revisionId, settings: {},
      })),
    },
    problems: [],
  };
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

/** Whether two requests ask for the same comparison, whatever order their keys are in. */
export function sameComparisonRequest(first: GenerationExperimentRequest, second: GenerationExperimentRequest): boolean {
  return canonical(first) === canonical(second);
}

interface ShapeCapability { available: boolean; preset_ids: OutputRatioPresetId[] }

/** The shapes both chosen workflows can be shown to make. */
export function sharedPresetIds(first: ShapeCapability | undefined, second: ShapeCapability | undefined): OutputRatioPresetId[] {
  if (!first?.available || !second?.available) return [];
  return first.preset_ids.filter((presetId) => second.preset_ids.includes(presetId));
}

export function trialIsWorking(status: TrialWorkStatus | null): boolean {
  return status === "queued" || status === "running" || status === "paused" || status === "blocked";
}

export function comparisonIsWorking(experiment: GenerationExperiment | undefined): boolean {
  return Boolean(experiment && experiment.state === "started"
    && experiment.arms.some((arm) => arm.trials.some((trial) => trialIsWorking(trial.status))));
}

/** The picture a finished trial made, read from its own run; null unless the run is that trial's. */
export function keptPicture(run: Run, trial: ExperimentTrial): string | null {
  const provenance = run.provenance_json as Record<string, unknown>;
  const witness = provenance.generation_experiment as Record<string, unknown> | undefined;
  if (run.id !== trial.run_id || run.work_step_id !== trial.work_step_id || witness?.trial_id !== trial.id) return null;
  const outputs = provenance.outputs;
  if (!Array.isArray(outputs)) return null;
  for (const output of outputs) {
    if (!output || typeof output !== "object") continue;
    const record = output as Record<string, unknown>;
    const origin = record.output_origin as Record<string, unknown> | undefined;
    // A preview the engine does not keep is never the picture shown.
    if (origin?.state === "attributed" && origin.output_type === "temp") continue;
    if (record.kind === "image" && typeof record.artifact_id === "string" && record.artifact_id) return record.artifact_id;
  }
  return null;
}

export interface SettingRow { key: string; values: [string, string]; differs: boolean }

function shown(value: unknown): string {
  if (value === undefined) return "not set";
  return typeof value === "string" ? value : JSON.stringify(value);
}

/** Each setting either choice resolved to, side by side; the shared negative prompt is left out. */
export function settingRows(first: Pick<ExperimentArm, "effective_settings">, second: Pick<ExperimentArm, "effective_settings">): SettingRow[] {
  const keys = [...new Set([...Object.keys(first.effective_settings), ...Object.keys(second.effective_settings)])]
    .filter((key) => key !== "negative_prompt").sort();
  return keys.map((key) => {
    const values: [string, string] = [shown(first.effective_settings[key]), shown(second.effective_settings[key])];
    return { key, values, differs: values[0] !== values[1] };
  });
}

export function refusalSubject(refusal: Pick<ExperimentRefusal, "arm_ordinal">, labels: string[]): string {
  if (refusal.arm_ordinal === 1 || refusal.arm_ordinal === 2) return labels[refusal.arm_ordinal - 1] ?? `Choice ${refusal.arm_ordinal}`;
  return "Whole comparison";
}

export type ComparisonNext = "check-again" | "read-again" | "confirm" | "retry" | "gone";

export interface ComparisonFailure {
  message: string;
  code: string | null;
  refusals: ExperimentRefusal[];
  estimate: ResourceEvidence[];
  next: ComparisonNext;
}

function refusalsFrom(value: unknown): ExperimentRefusal[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is ExperimentRefusal => Boolean(item) && typeof item === "object"
    && typeof (item as Record<string, unknown>).code === "string"
    && typeof (item as Record<string, unknown>).message === "string");
}

function estimateFrom(value: unknown): ResourceEvidence[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is ResourceEvidence => Boolean(item) && typeof item === "object"
    && typeof (item as Record<string, unknown>).value === "number"
    && ((item as Record<string, unknown>).resource === "work_units" || (item as Record<string, unknown>).resource === "output_bytes"));
}

/** What a failed comparison request means for the person, read from the error the client threw. */
export function comparisonFailure(error: unknown): ComparisonFailure {
  const record = error && typeof error === "object" ? error as Record<string, unknown> : {};
  const message = error instanceof Error ? error.message : "The comparison request failed.";
  const code = typeof record.code === "string" ? record.code : null;
  const payload = record.payload && typeof record.payload === "object" ? record.payload as Record<string, unknown> : {};
  const failure = { message, code, refusals: refusalsFrom(payload.refusals), estimate: estimateFrom(payload.estimate) };
  switch (code) {
    case "generation-experiment-refused":
    case "generation-experiment-preflight-changed":
      return { ...failure, next: "check-again" };
    case "generation-experiment-snapshot-changed":
    case "generation-experiment-already-started":
    case "generation-experiment-idempotency-conflict":
      return { ...failure, next: "read-again" };
    case "generation-experiment-confirmation-required":
      return { ...failure, next: "confirm" };
    case "generation-experiment-not-found":
    case "generation-experiment-record-invalid":
      return { ...failure, next: "gone" };
    case "request-validation-invalid":
      return { ...failure, message: "This comparison could not be sent as written.", next: "check-again" };
    default:
      return { ...failure, next: "retry" };
  }
}

/** A recipe draft as the recipe editor starts from it: only values a recipe can hold, never a default. */
export function recipeDraftPayload(draft: GenerationExperimentRecipeDraft): WorkflowUseCasePresetCreate {
  return {
    name: draft.name,
    use_case: draft.use_case,
    settings_json: Object.fromEntries(
      Object.entries(draft.settings_json).filter((entry): entry is [string, RecipeSettingValue] => isRecipeSettingValue(entry[1])),
    ),
    enabled: true,
    is_default: false,
  };
}
