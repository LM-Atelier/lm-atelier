import type { MessagePart } from "./types";

/** What the tool recorded when it compared the picture against the request.
 *
 * Read from the MESSAGE PART rather than from the artifact, and that is not an
 * implementation detail. Artifacts are addressed by their contents, so two runs
 * that produce identical bytes share one stored artifact - while whether a
 * picture is the size somebody asked for is a fact about the asking. Reading it
 * from the shared row would show one conversation the other's answer.
 *
 * Written beside the measurement when a picture is stored. Most values are not
 * judgements at all: the file may have been a preview node's throwaway, the
 * request may not have named a size, or the tool may not have been able to
 * confirm that the workflow's width and height really decided the output. Those
 * are recorded by name precisely so a reader cannot mistake "we did not look"
 * for "we looked and it was fine".
 */
export interface OutputSizeDisagreement {
  requestedWidth: number;
  requestedHeight: number;
  actualWidth: number;
  actualHeight: number;
}

function wholePixels(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : null;
}

/** The disagreement worth telling someone about, or nothing at all.
 *
 * Deliberately the only export that reads the record. Every state except
 * `disagreed` must stay silent, and there are six of them against the one - so
 * a caller writing its own `!== "agreed"` would warn on all six, including the
 * ones that mean the tool never formed an opinion. Returning the numbers rather
 * than a boolean keeps the caller from having to reach back into the record for
 * them and re-derive which is which.
 */
export function sizeDisagreement(part: MessagePart | null | undefined): OutputSizeDisagreement | null {
  const record = part?.metadata_json?.output_size_agreement;
  if (typeof record !== "object" || record === null) return null;
  const fields = record as Record<string, unknown>;
  if (fields.state !== "disagreed") return null;

  const requestedWidth = wholePixels(fields.requested_width);
  const requestedHeight = wholePixels(fields.requested_height);
  const actualWidth = wholePixels(fields.raster_width);
  const actualHeight = wholePixels(fields.raster_height);
  // A record that says "disagreed" without four usable numbers cannot be
  // rendered into a sentence anybody could act on, and a half-filled warning
  // beside a picture is worse than none - so it stays silent rather than
  // guessing at the missing half.
  if (
    requestedWidth === null
    || requestedHeight === null
    || actualWidth === null
    || actualHeight === null
  ) {
    return null;
  }
  return { requestedWidth, requestedHeight, actualWidth, actualHeight };
}

/** The sentence shown beside the picture.
 *
 * Says what happened and what was asked for, and stops. It does not apologise,
 * blame the workflow, or suggest a fix: the tool knows the two sizes differ and
 * genuinely does not know why, and a guess dressed as an explanation would be
 * worse than the bare fact.
 */
export function sizeDisagreementMessage(disagreement: OutputSizeDisagreement): string {
  const { requestedWidth, requestedHeight, actualWidth, actualHeight } = disagreement;
  return `This came out ${actualWidth} × ${actualHeight}, not the ${requestedWidth} × ${requestedHeight} you asked for.`;
}
