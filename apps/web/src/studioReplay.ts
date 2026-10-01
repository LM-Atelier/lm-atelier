/** The same edit again, with a new random seed: what Try another sends.
 *
 * Read from the run that made the result, not from how the Studio is set up
 * now, which may have moved on since. The settings are the ones that run
 * resolved, less what belongs to that one result: its seed, and how many it
 * made. The seed goes back as -1, which the server replaces with a fresh one,
 * so a profile that fixes a seed cannot make "another" the same picture. The
 * selection stays, because the picture it was drawn on is the one edited
 * again, and so do any second pictures the edit was given, such as a light
 * map or the picture a subject was taken from.
 */

import type { Run } from "./types";

export type StudioReplay = {
  words: string;
  /** The pictures the edit was given, the one it changed first. */
  inputs: string[];
  settings: Record<string, unknown>;
  workflowRevisionId?: string;
  /** The run was an enlargement: said again, since its settings may hold no factor to say it. */
  upscale?: boolean;
};

/** What a single result had to itself besides its seed, which is replaced rather than dropped. */
const PER_RESULT_SETTINGS = new Set(["noise_seed", "batch_size"]);

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

/** The replay of a run's edit, or nothing when the run did not record what it resolved. */
export function studioReplay(run: Run, words: string, inputs: string[]): StudioReplay | null {
  const resolved = record(run.provenance_json.resolved_settings);
  if (!resolved || inputs.length === 0) return null;
  const settings = Object.fromEntries(Object.entries(resolved).filter(([key]) => !PER_RESULT_SETTINGS.has(key)));
  const recorded = record(run.provenance_json.workflow)?.revision_id;
  const workflowRevisionId = typeof recorded === "string" && recorded ? recorded : run.workflow_revision_id;
  return {
    words,
    inputs,
    settings: { ...settings, seed: -1 },
    ...(workflowRevisionId ? { workflowRevisionId } : {}),
    ...(run.provenance_json.upscale === true ? { upscale: true } : {}),
  };
}
