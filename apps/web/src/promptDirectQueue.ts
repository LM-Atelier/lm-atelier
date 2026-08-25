import type { PromptBatch } from "./types";

export const MAX_PROMPT_BATCH_ITEMS = 16;
export const MAX_PROMPT_SELECTION_SEED = 2_147_483_647;

const MAX_IDENTIFIER_CHARACTERS = 40;
const MAX_PROMPT_CHARACTERS = 32_000;
const SHA256 = /^[0-9a-f]{64}$/;
const BATCH_KEYS = [
  "id", "chat_id", "prompt_template_id", "prompt_template_revision_id",
  "schema_version", "contract_sha256", "codec_version", "requested_count",
  "selection_seed", "plan_sha256", "state", "plan_version", "items", "replayed",
  "queue_idempotency_key", "work_plan_id", "queued_at",
] as const;
const ITEM_KEYS = [
  "id", "ordinal", "rendered_prompt", "rendered_sha256", "reviewed_prompt",
  "reviewed_sha256", "selected", "review_version", "reroll_count",
  "work_step_id", "run_id", "media_seed",
] as const;

export interface PromptDirectQueueAuthority {
  chatId: string;
  templateId: string;
  revisionId: string;
  schemaVersion: number;
  contractSha256: string;
  itemCount: number;
  selectionSeed: number;
  queueIdempotencyKey: string;
}

export class PromptDirectQueueAdmissionError extends Error {
  constructor() {
    super("Prompt batch response was invalid.");
    this.name = "PromptDirectQueueAdmissionError";
  }
}

function invalidBatch(): never {
  throw new PromptDirectQueueAdmissionError();
}

function exactRecord(value: unknown, keys: readonly string[]): Record<string, unknown> {
  if (
    typeof value !== "object"
    || value === null
    || Array.isArray(value)
    || Object.getPrototypeOf(value) !== Object.prototype
    || Object.getOwnPropertySymbols(value).length !== 0
  ) invalidBatch();
  const record = value as Record<string, unknown>;
  const actual = Object.keys(record);
  if (actual.length !== keys.length || keys.some((key) => !Object.hasOwn(record, key))) {
    invalidBatch();
  }
  return record;
}

function boundedString(value: unknown, maximum: number): string {
  if (
    typeof value !== "string"
    || value.length < 1
    || value.length > maximum
    || value.includes("\0")
  ) invalidBatch();
  for (let index = 0; index < value.length; index += 1) {
    const unit = value.charCodeAt(index);
    if (unit >= 0xd800 && unit <= 0xdbff) {
      const following = value.charCodeAt(index + 1);
      if (!(following >= 0xdc00 && following <= 0xdfff)) invalidBatch();
      index += 1;
    } else if (unit >= 0xdc00 && unit <= 0xdfff) {
      invalidBatch();
    }
  }
  return value;
}

function exactInteger(value: unknown, minimum: number, maximum: number): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    invalidBatch();
  }
  return value as number;
}

function digestString(value: unknown): string {
  const digest = boundedString(value, 64);
  if (!SHA256.test(digest)) invalidBatch();
  return digest;
}

function optionalIdentifier(value: unknown): string | null {
  return value === null ? null : boundedString(value, MAX_IDENTIFIER_CHARACTERS);
}

async function promptDigest(prompt: string): Promise<string> {
  const material = new TextEncoder().encode(`prompt-expansion-rendered-v1\0${prompt}`);
  const digest = await crypto.subtle.digest("SHA-256", material);
  return Array.from(
    new Uint8Array(digest),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
}

async function parsePromptBatch(
  value: unknown,
  authority: PromptDirectQueueAuthority,
): Promise<PromptBatch> {
  const raw = exactRecord(value, BATCH_KEYS);
  const id = boundedString(raw.id, MAX_IDENTIFIER_CHARACTERS);
  const chatId = boundedString(raw.chat_id, MAX_IDENTIFIER_CHARACTERS);
  const templateId = boundedString(raw.prompt_template_id, MAX_IDENTIFIER_CHARACTERS);
  const revisionId = boundedString(raw.prompt_template_revision_id, MAX_IDENTIFIER_CHARACTERS);
  const schemaVersion = exactInteger(raw.schema_version, 1, 1);
  const contractSha256 = digestString(raw.contract_sha256);
  const codecVersion = exactInteger(raw.codec_version, 2, 2);
  const requestedCount = exactInteger(raw.requested_count, 1, MAX_PROMPT_BATCH_ITEMS);
  const selectionSeed = exactInteger(
    raw.selection_seed,
    0,
    MAX_PROMPT_SELECTION_SEED,
  );
  const planSha256 = digestString(raw.plan_sha256);
  const state = boundedString(raw.state, 16);
  const planVersion = exactInteger(raw.plan_version, 1, Number.MAX_SAFE_INTEGER);
  const queueIdempotencyKey = raw.queue_idempotency_key === null
    ? null
    : boundedString(raw.queue_idempotency_key, 200);
  const workPlanId = optionalIdentifier(raw.work_plan_id);
  const queuedAt = raw.queued_at === null ? null : boundedString(raw.queued_at, 64);
  if (
    typeof raw.replayed !== "boolean"
    || (state !== "draft" && state !== "queued")
    || chatId !== authority.chatId
    || templateId !== authority.templateId
    || revisionId !== authority.revisionId
    || schemaVersion !== authority.schemaVersion
    || contractSha256 !== authority.contractSha256
    || requestedCount !== authority.itemCount
    || selectionSeed !== authority.selectionSeed
  ) invalidBatch();
  if (
    !Array.isArray(raw.items)
    || Object.getPrototypeOf(raw.items) !== Array.prototype
    || raw.items.length !== requestedCount
  ) invalidBatch();

  const itemIds = new Set<string>();
  const items = await Promise.all(raw.items.map(async (candidate, index) => {
    const entry = exactRecord(candidate, ITEM_KEYS);
    const itemId = boundedString(entry.id, MAX_IDENTIFIER_CHARACTERS);
    const ordinal = exactInteger(entry.ordinal, 1, MAX_PROMPT_BATCH_ITEMS);
    const renderedPrompt = boundedString(entry.rendered_prompt, MAX_PROMPT_CHARACTERS);
    const renderedSha256 = digestString(entry.rendered_sha256);
    const reviewedPrompt = boundedString(entry.reviewed_prompt, MAX_PROMPT_CHARACTERS);
    const reviewedSha256 = digestString(entry.reviewed_sha256);
    const reviewVersion = exactInteger(entry.review_version, 1, Number.MAX_SAFE_INTEGER);
    const rerollCount = exactInteger(entry.reroll_count, 0, Number.MAX_SAFE_INTEGER);
    const workStepId = optionalIdentifier(entry.work_step_id);
    const runId = optionalIdentifier(entry.run_id);
    const mediaSeed = entry.media_seed === null
      ? null
      : exactInteger(entry.media_seed, 0, MAX_PROMPT_SELECTION_SEED);
    if (
      ordinal !== index + 1
      || itemIds.has(itemId)
      || entry.selected !== true
      || reviewVersion !== 1
      || rerollCount !== 0
      || renderedPrompt !== reviewedPrompt
      || renderedSha256 !== reviewedSha256
    ) invalidBatch();
    itemIds.add(itemId);
    const computedDigest = await promptDigest(renderedPrompt);
    if (renderedSha256 !== computedDigest) invalidBatch();
    return {
      id: itemId,
      ordinal,
      rendered_prompt: renderedPrompt,
      rendered_sha256: renderedSha256,
      reviewed_prompt: reviewedPrompt,
      reviewed_sha256: reviewedSha256,
      selected: true,
      review_version: reviewVersion,
      reroll_count: rerollCount,
      work_step_id: workStepId,
      run_id: runId,
      media_seed: mediaSeed,
    };
  }));

  if (state === "draft") {
    if (
      planVersion !== 1
      || queueIdempotencyKey !== null
      || workPlanId !== null
      || queuedAt !== null
      || items.some((item) =>
        item.work_step_id !== null || item.run_id !== null || item.media_seed !== null)
    ) invalidBatch();
  } else {
    if (
      planVersion !== 2
      || queueIdempotencyKey !== authority.queueIdempotencyKey
      || workPlanId === null
      || queuedAt === null
      || Number.isNaN(Date.parse(queuedAt))
    ) invalidBatch();
    const workStepIds = new Set<string>();
    const runIds = new Set<string>();
    const mediaSeeds = new Set<number>();
    for (const item of items) {
      if (
        item.work_step_id === null
        || item.run_id === null
        || item.media_seed === null
        || workStepIds.has(item.work_step_id)
        || runIds.has(item.run_id)
        || mediaSeeds.has(item.media_seed)
      ) invalidBatch();
      workStepIds.add(item.work_step_id);
      runIds.add(item.run_id);
      mediaSeeds.add(item.media_seed);
    }
  }

  return {
    id,
    chat_id: chatId,
    prompt_template_id: templateId,
    prompt_template_revision_id: revisionId,
    schema_version: schemaVersion,
    contract_sha256: contractSha256,
    codec_version: codecVersion as 2,
    requested_count: requestedCount,
    selection_seed: selectionSeed,
    plan_sha256: planSha256,
    state,
    plan_version: planVersion,
    items,
    replayed: raw.replayed,
    queue_idempotency_key: queueIdempotencyKey,
    work_plan_id: workPlanId,
    queued_at: queuedAt,
  };
}

export async function admitCreatedPromptBatch(
  value: unknown,
  authority: PromptDirectQueueAuthority,
): Promise<PromptBatch> {
  const admitted = await parsePromptBatch(value, authority);
  if (admitted.state === "queued" && !admitted.replayed) invalidBatch();
  return admitted;
}

export async function admitQueuedPromptBatch(
  value: unknown,
  authority: PromptDirectQueueAuthority,
  draft: PromptBatch,
): Promise<PromptBatch> {
  const admitted = await parsePromptBatch(value, authority);
  if (
    admitted.state !== "queued"
    || admitted.id !== draft.id
    || admitted.plan_version !== draft.plan_version + 1
    || admitted.plan_sha256 !== draft.plan_sha256
    || admitted.items.length !== draft.items.length
    || admitted.items.some((item, index) => {
      const previous = draft.items[index];
      return item.id !== previous.id
        || item.ordinal !== previous.ordinal
        || item.rendered_prompt !== previous.rendered_prompt
        || item.rendered_sha256 !== previous.rendered_sha256
        || item.reviewed_prompt !== previous.reviewed_prompt
        || item.reviewed_sha256 !== previous.reviewed_sha256
        || item.selected !== previous.selected
        || item.review_version !== previous.review_version
        || item.reroll_count !== previous.reroll_count;
    })
  ) invalidBatch();
  return admitted;
}

export function promptCountMaximum(maximum: number): number {
  if (!Number.isFinite(maximum)) return 1;
  return Math.max(1, Math.min(MAX_PROMPT_BATCH_ITEMS, Math.floor(maximum)));
}

export function randomPromptSelectionSeed(): number {
  const value = new Uint32Array(1);
  crypto.getRandomValues(value);
  return value[0] & MAX_PROMPT_SELECTION_SEED;
}
