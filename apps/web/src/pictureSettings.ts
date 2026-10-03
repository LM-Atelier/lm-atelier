/** What a picture's own file says about how it was made, as the server read it. */
export interface PictureSettings {
  dialect: "parameters" | "comfyui_prompt" | "none";
  claims: { key: string; value: string | number; source: string }[];
  ignored: { name: string; reason: string }[];
  warnings: string[];
}

const DIALECTS = new Set(["parameters", "comfyui_prompt", "none"]);

/** The settings a claim can name, in the order they are shown. */
const SETTING_LABELS: Record<string, string> = {
  prompt: "Prompt",
  negative_prompt: "Negative prompt",
  seed: "Seed",
  steps: "Steps",
  guidance: "Guidance",
  sampler: "Sampler",
  scheduler: "Scheduler",
  denoise: "Denoise",
  width: "Width",
  height: "Height",
  batch: "Batch",
};

/** Why something in the file was left out, in words. */
const IGNORED_REASONS: Record<string, string> = {
  names_a_file: "names a file on the computer that made it",
  workflow_graph: "a workflow, which is never imported or run",
  not_used: "not a setting this reads",
  not_settings: "not settings",
  malformed: "not readable as that kind of value",
  unsafe_text: "holds characters that could change how text is shown",
  compressed: "compressed text, which is not expanded",
  xmp_not_read: "XMP text, which is not read",
  unsupported_encoding: "text in an encoding this does not read",
  not_text: "not readable text",
  too_long: "too long to read",
  not_utf8: "not readable text",
  from_another_node: "set by another part of the workflow",
  not_plain_text: "not plain text",
  repeated: "given more than once",
  trailing_text: "text after the settings line",
  too_complex: "too complex to read",
  settings_unrecognized: "no settings line",
  no_sampler: "no sampler to read settings from",
  several_samplers: "more than one sampler, so no one set of settings",
  empty: "empty",
};

/** What is said about the file as a whole, by the reader's warning, first match first. */
const WARNING_NOTES: [string, string][] = [
  ["format_not_read", "Only PNG, JPEG and WebP pictures are read for settings."],
  ["too_large", "This picture is too large to read for settings."],
  ["several_samplers", "This picture's workflow has more than one sampler, so no one set of settings is shown."],
  ["no_sampler", "This picture's workflow has no sampler to read settings from."],
  ["settings_unrecognized", "The settings text in this picture has no line of settings."],
];

function record(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("picture settings are not readable");
  }
  return value as Record<string, unknown>;
}

function text(value: unknown): string {
  if (typeof value !== "string") throw new Error("picture settings are not readable");
  return value;
}

function list(value: unknown): unknown[] {
  if (!Array.isArray(value)) throw new Error("picture settings are not readable");
  return value;
}

/** Read the server's answer, refusing one of another shape rather than showing part of it. */
export function readPictureSettings(value: unknown): PictureSettings {
  const answer = record(value);
  const dialect = text(answer.dialect);
  if (!DIALECTS.has(dialect)) throw new Error("picture settings are not readable");
  return {
    dialect: dialect as PictureSettings["dialect"],
    claims: list(answer.claims).map((item) => {
      const claim = record(item);
      const claimValue = claim.value;
      // A whole number past what a number holds exactly comes as text; one that did not was rounded.
      if (typeof claimValue !== "string" && (
        typeof claimValue !== "number"
        || !Number.isFinite(claimValue)
        || (Number.isInteger(claimValue) && !Number.isSafeInteger(claimValue))
      )) {
        throw new Error("picture settings are not readable");
      }
      return { key: text(claim.key), value: claimValue, source: text(claim.source) };
    }),
    ignored: list(answer.ignored).map((item) => {
      const entry = record(item);
      return { name: text(entry.name), reason: text(entry.reason) };
    }),
    warnings: list(answer.warnings).map(text),
  };
}

export function settingLabel(key: string): string {
  return SETTING_LABELS[key] ?? key;
}

export function ignoredReason(reason: string): string {
  return IGNORED_REASONS[reason] ?? "left out";
}

/** The one line said about the file as a whole, when there are no settings to list. */
export function pictureSettingsNote(settings: PictureSettings): string | null {
  const warned = WARNING_NOTES.find(([warning]) => settings.warnings.includes(warning));
  if (warned) return warned[1];
  if (settings.claims.length > 0) return null;
  return settings.dialect === "none"
    ? "No settings were found in this picture's file."
    : "No settings could be read from this picture's file.";
}

/** Whether the list of what was left out stopped at its bound. */
export function pictureSettingsCut(settings: PictureSettings): boolean {
  return settings.warnings.includes("too_many_entries");
}
