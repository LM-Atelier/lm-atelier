/**
 * Reading a generation record for display.
 *
 * The record is downloaded exactly as the server wrote it, so these helpers only
 * read it: what it says, what it left out and what it falls short of for a
 * replay, in words a person can act on. They never change or re-encode it.
 */

export const GENERATION_RECORD_SCHEMA = "lm-atelier-output-recipe-v1";

export type GenerationRecordSummary = {
  digest: string;
  outputSha256: string;
  kind: "image" | "video";
  operation: string;
  promptIncluded: boolean;
  promptOmittedReason: string | null;
  seed: number | null;
  seedBinding: string;
  settingCount: number;
  inputCount: number;
  workflowVerified: boolean | null;
  modelFileCount: number | null;
  loraCount: number;
  removed: string[];
  missing: string[];
};

const OPERATION_TEXT: Record<string, string> = {
  text_to_image: "Picture from a description",
  image_to_image: "Picture edit",
  text_to_video: "Video from a description",
  image_to_video: "Video from a picture",
};

const OMISSION_TEXT: Record<string, string> = {
  chosen: "Left out, as you chose.",
  removed_from_chat: "Left out because a message it could repeat was removed from the chat.",
  contains_local_reference: "Left out because it names a file or item on this computer.",
  too_long: "Left out because it is too long for a record.",
  unavailable: "Left out because it could not be read reliably.",
};

const MISSING_TEXT: Record<string, string> = {
  prompt_omitted: "The prompt is not included.",
  frozen_snapshot_absent: "This generation did not keep a frozen copy of its inputs.",
  frozen_snapshot_unavailable: "Its frozen copy of its inputs can no longer be read.",
  depends_on_other_outputs: "It used another generation's output, which is not in this record.",
  finished_after_generation: "The picture was finished after the workflow ran, so a replay of the workflow alone would differ.",
  seed_not_recorded: "The seed was not recorded.",
  workflow_unavailable: "The workflow it ran is not available.",
  workflow_unverified: "The workflow's identity could not be confirmed.",
  model_files_not_recorded: "The model's files were not recorded.",
  lora_identity_missing: "An added LoRA has no recorded file identity.",
  settings_removed: "Some settings were left out.",
  input_unavailable: "An input picture is no longer stored here.",
  mock_engine: "It was made by the test engine, not a real model.",
};

const REMOVED_TEXT: Record<string, string> = {
  prompt: "The prompt",
  settings: "A setting with an unusable name",
  "settings.mask": "How the selection was applied",
  "settings.workflow_lora_overrides": "Changes to the workflow's own LoRAs",
  "loras.names": "LoRA names and trigger words",
  inputs: "An input that is not a stored picture",
  "model.files": "Some model file names",
  "model.provider": "The model's source",
  "model.remote_id": "The model's source name",
  "model.revision": "The model's source version",
  "model.content_rating": "The model's content rating",
};

export function operationText(operation: string): string {
  return OPERATION_TEXT[operation] ?? "Generation";
}

export function omissionText(reason: string | null): string {
  return (reason && OMISSION_TEXT[reason]) || "Left out.";
}

export function missingText(reason: string): string {
  return MISSING_TEXT[reason] ?? "Something this record cannot vouch for.";
}

export function removedText(name: string): string {
  if (REMOVED_TEXT[name]) return REMOVED_TEXT[name];
  if (name.startsWith("settings.")) return `The setting "${name.slice("settings.".length)}"`;
  return name;
}

export function generationRecordFileName(summary: Pick<GenerationRecordSummary, "outputSha256">): string {
  return `generation-record-${summary.outputSha256.slice(0, 12)}.json`;
}

export function generationRecordBundleFileName(summary: Pick<GenerationRecordSummary, "outputSha256">): string {
  return `generation-record-${summary.outputSha256.slice(0, 12)}.zip`;
}

function object(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("The generation record is malformed.");
  }
  return value as Record<string, unknown>;
}

function names(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

/** What a downloaded record says, read without altering a byte of it. */
export function readGenerationRecord(bytes: ArrayBuffer): GenerationRecordSummary {
  const record = object(JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)));
  if (record.schema !== GENERATION_RECORD_SCHEMA || record.version !== 1) {
    throw new Error("This is not a generation record this version can read.");
  }
  const output = object(record.output);
  const prompt = object(record.prompt);
  const seed = object(record.seed);
  const settings = object(record.settings);
  const workflow = record.workflow === null ? null : object(record.workflow);
  const model = record.model === null ? null : object(record.model);
  const reproducibility = object(record.reproducibility);
  if (typeof record.digest !== "string" || typeof output.sha256 !== "string") {
    throw new Error("The generation record is malformed.");
  }
  return {
    digest: record.digest,
    outputSha256: output.sha256,
    kind: output.kind === "video" ? "video" : "image",
    operation: typeof record.operation === "string" ? record.operation : "",
    promptIncluded: prompt.included === true,
    promptOmittedReason: typeof prompt.omitted_reason === "string" ? prompt.omitted_reason : null,
    seed: typeof seed.value === "number" ? seed.value : null,
    seedBinding: typeof seed.binding === "string" ? seed.binding : "not_recorded",
    settingCount: Object.keys(object(settings.bound)).length + Object.keys(object(settings.unbound)).length,
    inputCount: Array.isArray(record.inputs) ? record.inputs.length : 0,
    workflowVerified: workflow === null ? null : workflow.verified === true,
    modelFileCount: model === null ? null : Object.keys(object(model.files)).length,
    loraCount: Array.isArray(record.loras) ? record.loras.length : 0,
    removed: names(record.removed),
    missing: names(reproducibility.missing),
  };
}

export type RequirementKind = "workflow" | "model_file" | "lora" | "input";
export type RequirementState = "present" | "inactive" | "missing";

export type GenerationRecordRequirement = {
  kind: RequirementKind;
  sha256: string;
  role: string | null;
  state: RequirementState;
};

export type GenerationRecordCheck = {
  digest: string;
  operation: string;
  requirements: GenerationRecordRequirement[];
  allPresent: boolean;
  missing: string[];
  /** The copy of the picture a bundle holds beside its record; null for a bare record. */
  picture: { width: number; height: number } | null;
};

const KIND_TEXT: Record<RequirementKind, string> = {
  workflow: "Workflow",
  model_file: "Model file",
  lora: "LoRA",
  input: "Input picture",
};

const STATE_TEXT: Record<RequirementState, string> = {
  present: "Here and ready",
  inactive: "Here, not ready",
  missing: "Not here",
};

export function requirementKindText(kind: RequirementKind): string {
  return KIND_TEXT[kind];
}

export function requirementStateText(state: RequirementState): string {
  return STATE_TEXT[state];
}

const KINDS = new Set<string>(Object.keys(KIND_TEXT));
const STATES = new Set<string>(Object.keys(STATE_TEXT));
const HEX64 = /^[0-9a-f]{64}$/;

/** The check's answer, read strictly: anything unexpected is refused rather than shown. */
export function readGenerationRecordCheck(value: unknown): GenerationRecordCheck {
  const report = object(value);
  if (typeof report.digest !== "string" || typeof report.operation !== "string" || !Array.isArray(report.requirements)) {
    throw new Error("The check's answer is malformed.");
  }
  const requirements = report.requirements.map((item): GenerationRecordRequirement => {
    const entry = object(item);
    if (
      typeof entry.kind !== "string" || !KINDS.has(entry.kind)
      || typeof entry.state !== "string" || !STATES.has(entry.state)
      || typeof entry.sha256 !== "string" || !HEX64.test(entry.sha256)
      || (entry.role !== null && typeof entry.role !== "string")
    ) {
      throw new Error("The check's answer is malformed.");
    }
    return {
      kind: entry.kind as RequirementKind,
      sha256: entry.sha256,
      role: entry.role as string | null,
      state: entry.state as RequirementState,
    };
  });
  const reproducibility = object(report.reproducibility);
  return {
    digest: report.digest,
    operation: report.operation,
    requirements,
    allPresent: report.all_present === true && requirements.every((item) => item.state === "present"),
    missing: names(reproducibility.missing),
    picture: checkedPicture(report.picture),
  };
}

function checkedPicture(value: unknown): GenerationRecordCheck["picture"] {
  if (value === null || value === undefined) return null;
  const picture = object(value);
  const whole = (side: unknown): side is number => Number.isInteger(side) && (side as number) > 0;
  if (
    typeof picture.sha256 !== "string" || !HEX64.test(picture.sha256)
    || typeof picture.copy_of !== "string" || !HEX64.test(picture.copy_of)
    || !whole(picture.width) || !whole(picture.height)
  ) {
    throw new Error("The check's answer is malformed.");
  }
  return { width: picture.width, height: picture.height };
}

/** A chosen file's exact bytes, read the way every browser and test DOM supports. */
export function readFileBytes(file: Blob): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) resolve(reader.result);
      else reject(new Error("The file could not be read."));
    };
    reader.onerror = () => reject(new Error("The file could not be read."));
    reader.readAsArrayBuffer(file);
  });
}

/** What a replay starts: generation from words, a video from a picture, and an edit of a picture. */
export const REPLAYABLE_OPERATIONS = new Set(["text_to_image", "text_to_video", "image_to_video", "image_to_image"]);
/** The generations whose settings can be kept as a recipe; an edit's belong to its own picture. */
export const RECIPE_DRAFT_OPERATIONS = new Set(["text_to_image", "text_to_video", "image_to_video"]);

export type ReplayRefusal = { code: string; sha256: string | null; reasons: string[] };

export type ReplayPlan = {
  digest: string;
  operation: string;
  ready: boolean;
  refusals: ReplayRefusal[];
};

const REPLAY_REFUSAL_TEXT: Record<string, string> = {
  "replay-record-incomplete": "The record leaves out something a replay needs",
  "replay-record-unsupported": "The record asks for something a replay cannot do yet",
  "replay-engine-differs": "It was made with a different generation engine",
  "replay-workflow-missing": "Its workflow is not here exactly",
  "replay-workflow-ambiguous": "More than one workflow here matches it exactly",
  "replay-workflow-not-ready": "Its workflow is here but not ready to run",
  "replay-workflow-binding-differs": "Its workflow is set up with different files here",
  "replay-workflow-unscoped": "Its workflow does not say which files it used",
  "replay-model-missing": "No model here holds exactly its files",
  "replay-model-ambiguous": "More than one model here holds exactly its files",
  "replay-lora-missing": "One of its LoRAs is not here",
  "replay-lora-ambiguous": "One of its LoRAs is here more than once",
  "replay-lora-unusable": "Its LoRAs cannot be added to its workflow here",
  "replay-input-missing": "One of its input pictures is not here",
  "adaptation-workflow-unusable": "The chosen workflow cannot make this here",
  "adaptation-model-unusable": "The chosen model cannot be used for this here",
  "adaptation-lora-unusable": "A chosen LoRA is not here",
  "adaptation-input-unusable": "A chosen picture is not here, or stands in for two of its pictures",
};

/** What can be chosen in place of each requirement a new version may stand in for. */
const ADAPTABLE_REFUSALS: Record<string, "workflow" | "model" | "loras" | "inputs"> = {
  "replay-workflow-missing": "workflow",
  "replay-workflow-ambiguous": "workflow",
  "replay-workflow-not-ready": "workflow",
  "replay-workflow-binding-differs": "workflow",
  "replay-workflow-unscoped": "workflow",
  "replay-model-missing": "model",
  "replay-model-ambiguous": "model",
  "replay-lora-missing": "loras",
  "replay-lora-ambiguous": "loras",
  "replay-lora-unusable": "loras",
  "replay-input-missing": "inputs",
};

export type AdaptationNeeds = { workflow: boolean; model: boolean; loras: boolean; inputs: boolean };

/** What a new version needs chosen, or null when it cannot stand in for everything that refused. */
export function adaptationNeeds(plan: ReplayPlan): AdaptationNeeds | null {
  if (plan.ready || !REPLAYABLE_OPERATIONS.has(plan.operation)) return null;
  const needs = plan.refusals.map((refusal) => ADAPTABLE_REFUSALS[refusal.code]);
  if (needs.some((need) => need === undefined)) return null;
  return {
    workflow: needs.includes("workflow"),
    model: needs.includes("model"),
    loras: needs.includes("loras"),
    inputs: needs.includes("inputs"),
  };
}

/** Why a new version was not started, in fixed words; never the server's own text. */
export function adaptationFailureText(error: unknown): string {
  const failure = error as { code?: unknown; payload?: Record<string, unknown> } | null;
  const refusals = failure?.code === "adaptation-unavailable" && Array.isArray(failure.payload?.refusals)
    ? failure.payload.refusals.flatMap((item) => {
      const code = object(item).code;
      return typeof code === "string" ? [replayRefusalText(code)] : [];
    })
    : [];
  return refusals.length > 0
    ? `It cannot be made with these choices: ${refusals.join("; ")}. Nothing was started.`
    : "It could not be started here. Nothing was started.";
}

const REPLAY_REASON_TEXT: Record<string, string> = {
  workflow_contract_version: "Its workflow is described in another version's terms.",
  too_many_inputs: "It has more input pictures than one turn takes.",
  too_many_loras: "It has more LoRAs than one turn takes.",
  lora_strength: "A LoRA strength is out of range.",
  lora_positions: "Its LoRAs are out of order.",
  inputs_for_operation: "Its inputs do not fit how it was made.",
  mask_input: "Its selection cannot be sent again: only a picture edit's selection, sent as drawn with no feathering, turning round or blending, can be.",
  repeated_inputs: "It lists the same input picture twice.",
  edit_prompt_wording: "Its edit was worded by another version, so the request cannot be read back out.",
};

const SECTION_TEXT: Record<string, string> = {
  operation: "how it was made",
  prompt: "the prompt",
  seed: "the seed",
  settings: "the settings",
  inputs: "the input pictures",
  workflow: "the workflow",
  model: "the model",
  loras: "the LoRAs",
  output_count: "how many results it makes",
  completeness: "what could be recorded",
};

/** Sections named as a reader says them: "the model and the workflow". */
function sectionList(sections: string[]): string {
  const named = sections.map((section) => SECTION_TEXT[section] ?? section);
  return named.length > 1 ? `${named.slice(0, -1).join(", ")} and ${named[named.length - 1]}` : named.join("");
}

export function replayRefusalText(code: string): string {
  return REPLAY_REFUSAL_TEXT[code] ?? "Something it names does not match here";
}

export function replayReasonText(reason: string): string {
  return REPLAY_REASON_TEXT[reason] ?? missingText(reason);
}

/** Why starting a replay was refused, in fixed words; never the server's own text. */
export function replayFailureText(error: unknown): string {
  const failure = error as { code?: unknown; payload?: Record<string, unknown> } | null;
  if (failure?.code === "replay-differs") {
    const sections = Array.isArray(failure.payload?.sections)
      ? failure.payload.sections.filter((item): item is string => typeof item === "string")
      : [];
    const named = sectionList(sections);
    return `Generating it here would not match the record exactly${named ? ` (it would differ in ${named})` : ""}. Nothing was started.`;
  }
  return "It could not be started here. Nothing was started.";
}

/** What the replay plan says, read strictly; anything unexpected is refused whole. */
export function readReplayPlan(value: unknown): ReplayPlan {
  const plan = object(value);
  if (
    typeof plan.digest !== "string"
    || typeof plan.operation !== "string"
    || typeof plan.ready !== "boolean"
    || !Array.isArray(plan.refusals)
  ) {
    throw new Error("The replay plan is malformed.");
  }
  const refusals = plan.refusals.map((item): ReplayRefusal => {
    const refusal = object(item);
    if (
      typeof refusal.code !== "string"
      || (refusal.sha256 !== null && (typeof refusal.sha256 !== "string" || !HEX64.test(refusal.sha256)))
      || !Array.isArray(refusal.reasons)
    ) {
      throw new Error("The replay plan is malformed.");
    }
    return { code: refusal.code, sha256: refusal.sha256 as string | null, reasons: names(refusal.reasons) };
  });
  if (plan.ready === (refusals.length > 0)) throw new Error("The replay plan contradicts itself.");
  return { digest: plan.digest, operation: plan.operation, ready: plan.ready, refusals };
}

type ReplayState = "pending" | "identical" | "different" | "output_missing";
/** How a run made from a record turned out; a new version also says what differs from the record. */
export type ReplayOutcome = { state: ReplayState | "adapted"; differs: string[] };

const REPLAY_OUTCOME_TEXT: Record<ReplayState, string> = {
  pending: "Generated again from a record. It has not finished yet.",
  identical: "Generated again from a record, and it came out exactly as the recorded one did.",
  different:
    "Generated again from a record with everything it names matched, but it came out differently. "
    + "The record does not hold everything that can change a result, such as the runtime's version.",
  output_missing: "Generated again from a record, but it left no result to compare.",
};

/** What a run made from a record says about it, or null for a run that was not made from one. */
export function readReplayOutcome(value: unknown): ReplayOutcome | null {
  const outcome = object(value);
  if (outcome.state === "adapted") return { state: "adapted", differs: names(outcome.differs) };
  return typeof outcome.state === "string" && outcome.state in REPLAY_OUTCOME_TEXT
    ? { state: outcome.state as ReplayState, differs: [] }
    : null;
}

export function replayOutcomeText(outcome: ReplayOutcome): string {
  if (outcome.state !== "adapted") return REPLAY_OUTCOME_TEXT[outcome.state];
  return outcome.differs.length > 0
    ? `Made as a new version of a record, not that generation again. It differs from the record in ${sectionList(outcome.differs)}.`
    : "Made as a new version of a record, not that generation again, though nothing the record holds differs.";
}
