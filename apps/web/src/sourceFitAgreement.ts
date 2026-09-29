import type { MessagePart } from "./types";

/** A source mismatch belongs to this output occurrence, never the shared artifact. */
export function sourcePixelsChanged(part: MessagePart): boolean {
  const record = part.metadata_json.source_fit_agreement;
  if (typeof record !== "object" || record === null || Array.isArray(record)) return false;
  const fields = record as Record<string, unknown>;
  return fields.v === 1 && fields.state === "changed";
}
