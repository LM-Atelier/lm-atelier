/** A remix preview: a picture's own settings judged against a workflow and model chosen here. */

export type RemixClaimState = "supported" | "unresolved" | "incompatible" | "ignored";

export interface RemixPreviewClaim {
  key: string;
  setting: string | null;
  value: string | number;
  source: string;
  state: RemixClaimState;
  reason: string | null;
  applied: boolean;
}

export type RemixRole = "words" | "edit";

export interface RemixPreview {
  /** Made from words, or starting from the picture itself. */
  role: RemixRole;
  operation: "text_to_image" | "image_to_image";
  /** The picture a remix starts from, for a remix that starts from it. */
  source: { width: number; height: number } | null;
  workflow_revision_id: string;
  profile_id: string;
  claims: RemixPreviewClaim[];
  /** The picture's shape at a size this workflow makes, offered when its own size cannot be used. */
  shape: { width: number; height: number } | null;
  ready: boolean;
  refusals: { code: string; message: string }[];
  resolved: {
    text: string;
    settings: Record<string, unknown>;
    /** A seed is drawn when the picture is made, not one set here or the workflow's own. */
    seed_drawn: boolean;
    trigger_words: string[];
    /** The words the engine is given. */
    engine_prompt: string;
    /** How much a remix starting from the picture changes it, and where that came from. */
    strength: { parameter: string; mode: "auto" | "manual"; value: number; from_file: boolean } | null;
  } | null;
  review_digest: string | null;
}

const STATES = new Set<RemixClaimState>(["supported", "unresolved", "incompatible", "ignored"]);

/** Why a claim is not applied as it is, in words. */
const REASONS: Record<string, string> = {
  no_vocabulary: "this workflow does not list the names it knows, so this one cannot be checked",
  several_controls: "this workflow uses this setting in more than one place",
  size_incomplete: "only half of a size was given",
  size_unproven: "this workflow cannot show it makes this size",
  not_offered: "this workflow does not take this setting",
  value_refused: "this workflow does not take this value",
  not_a_choice: "this workflow does not know this name",
  size_not_offered: "this workflow does not make this size",
  size_differs: "the picture is not this size",
  replaces_picture: "it would draw the whole picture again",
  edit_only: "only for changing a picture, not making one from words",
  one_picture: "a remix makes one picture",
};

function fail(): never {
  throw new Error("The remix preview is not readable.");
}

function record(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) fail();
  return value as Record<string, unknown>;
}

function text(value: unknown): string {
  if (typeof value !== "string") fail();
  return value;
}

function optionalText(value: unknown): string | null {
  return value === null ? null : text(value);
}

function list(value: unknown): unknown[] {
  if (!Array.isArray(value)) fail();
  return value;
}

function claim(value: unknown): RemixPreviewClaim {
  const item = record(value);
  const claimValue = item.value;
  // A whole number past what a number holds exactly comes as text; one that did not was rounded.
  if (typeof claimValue !== "string" && (
    typeof claimValue !== "number"
    || !Number.isFinite(claimValue)
    || (Number.isInteger(claimValue) && !Number.isSafeInteger(claimValue))
  )) fail();
  const state = text(item.state);
  if (!STATES.has(state as RemixClaimState) || typeof item.applied !== "boolean") fail();
  return {
    key: text(item.key),
    setting: optionalText(item.setting),
    value: claimValue,
    source: text(item.source),
    state: state as RemixClaimState,
    reason: optionalText(item.reason),
    applied: item.applied,
  };
}

function size(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1) fail();
  return value;
}

function strength(value: unknown): NonNullable<RemixPreview["resolved"]>["strength"] {
  if (value === null) return null;
  const item = record(value);
  if (
    (item.mode !== "auto" && item.mode !== "manual")
    || typeof item.value !== "number" || !Number.isFinite(item.value)
    || typeof item.from_file !== "boolean"
  ) fail();
  return { parameter: text(item.parameter), mode: item.mode, value: item.value, from_file: item.from_file };
}

/** Read the server's answer, refusing one of another shape rather than showing part of it. */
export function readRemixPreview(value: unknown): RemixPreview {
  const answer = record(value);
  if (typeof answer.ready !== "boolean") fail();
  if (answer.role !== "words" && answer.role !== "edit") fail();
  if (answer.operation !== "text_to_image" && answer.operation !== "image_to_image") fail();
  const resolved = answer.resolved === null ? null : record(answer.resolved);
  if (resolved !== null && typeof resolved.seed_drawn !== "boolean") fail();
  return {
    role: answer.role,
    operation: answer.operation,
    source: answer.source === null ? null : (() => {
      const source = record(answer.source);
      return { width: size(source.width), height: size(source.height) };
    })(),
    workflow_revision_id: text(answer.workflow_revision_id),
    profile_id: text(answer.profile_id),
    claims: list(answer.claims).map(claim),
    shape: answer.shape === null ? null : (() => {
      const shape = record(answer.shape);
      return { width: size(shape.width), height: size(shape.height) };
    })(),
    ready: answer.ready,
    refusals: list(answer.refusals).map((item) => {
      const refusal = record(item);
      return { code: text(refusal.code), message: text(refusal.message) };
    }),
    resolved: resolved === null ? null : {
      text: text(resolved.text),
      settings: record(resolved.settings),
      seed_drawn: resolved.seed_drawn as boolean,
      trigger_words: list(resolved.trigger_words).map(text),
      engine_prompt: text(resolved.engine_prompt),
      strength: strength(resolved.strength),
    },
    review_digest: optionalText(answer.review_digest),
  };
}

/** What a remix does with one claim, in words. */
export function remixClaimText(item: RemixPreviewClaim): string {
  const reason = item.reason ? REASONS[item.reason] ?? "left out" : null;
  if (item.state === "supported") return "can be used as it is";
  if (item.state === "unresolved") return `cannot be checked: ${reason}`;
  if (item.state === "incompatible") return `cannot be used: ${reason}`;
  return `is not used: ${reason}`;
}

/**
 * The claims a person can choose to apply, with width and height as one size,
 * and the picture's shape when a size this workflow makes is offered for it.
 */
export function applicableClaims(
  claims: RemixPreviewClaim[],
  shape: RemixPreview["shape"] = null,
): string[] {
  // A claim with no setting is used as it is (the words, a kept size), never chosen.
  const keys = claims
    .filter((item) => item.state === "supported" && item.key !== "prompt" && item.setting !== null)
    .map((item) => item.key);
  // Half a size cannot be applied, so a lone width or height is not offered.
  const whole = keys.includes("width") && keys.includes("height");
  return [
    ...keys.filter((key) => key !== "width" && key !== "height"),
    ...(whole ? ["size"] : []),
    ...(shape ? ["shape"] : []),
  ];
}

/** What the person is told when a remix could not be started. */
const FAILURES: Record<string, string> = {
  "remix-review-changed": "What this remix would run changed since it was shown. It has been checked again.",
  "remix-differs": "This remix would not run as it was shown. Check it again.",
  "remix-unavailable": "This picture cannot be remixed with these choices.",
  "remix-chat-not-clean": "A remix is made only in a new chat with nothing in it.",
};

export function remixFailureText(error: unknown): string {
  const code = (error as { code?: unknown } | null)?.code;
  return typeof code === "string" && code in FAILURES
    ? FAILURES[code]
    : "This remix could not be started.";
}

/** The claim keys one choice stands for: a size is its width and its height. */
export function claimKeys(choice: string): string[] {
  return choice === "size" ? ["width", "height"] : [choice];
}
