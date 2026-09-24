/** A LoRA's trigger words: what its file declares, and what a person recorded.
 *
 * The two are stored apart and shown apart, but a run applies them together,
 * so this reads them in the same order the server does.
 */

import type { ModelAssetInstall } from "./types";

type TriggerWordSource = Pick<ModelAssetInstall, "manifest_json" | "typed_trigger_words">;

function words(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((word): word is string => typeof word === "string") : [];
}

/** Keep each word once in any casing, first spelling wins, blanks dropped. */
function unique(values: string[]): string[] {
  const seen = new Set<string>();
  const kept: string[] = [];
  for (const value of values) {
    const word = value.trim();
    const key = word.toLowerCase();
    if (!word || seen.has(key)) continue;
    seen.add(key);
    kept.push(word);
  }
  return kept;
}

/** What the LoRA's own file declares: its trigger words, then its trained words. */
export function measuredTriggerWords(asset: Pick<ModelAssetInstall, "manifest_json">): string[] {
  const metadata = asset.manifest_json.metadata;
  if (!metadata || typeof metadata !== "object") return [];
  const record = metadata as Record<string, unknown>;
  return unique([...words(record.trigger_words), ...words(record.trained_words)]);
}

/** Every word a run adds for this LoRA: the file's first, then the recorded ones. */
export function loraTriggerWords(asset: TriggerWordSource): string[] {
  return unique([...measuredTriggerWords(asset), ...words(asset.typed_trigger_words)]);
}

/** Read what a person typed as words: commas separate them. */
export function parseTypedTriggerWords(text: string): string[] {
  return unique(text.split(","));
}
