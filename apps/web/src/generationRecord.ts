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
