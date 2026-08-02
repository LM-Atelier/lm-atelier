import type { Message } from "./types";

/** The workshop transcript without the dialog's own plumbing.
 *
 * The opening instruction is sent programmatically before anything renders.
 * Showing it spoils the effect - the improved prompt should simply appear -
 * so the leading user message is dropped. Everything the user types stays.
 */
export function workshopTranscript(messages: Message[]): Message[] {
  return messages[0]?.role === "user" ? messages.slice(1) : messages;
}

/** Say which kind of help this was: a rewrite grounded in the actual image,
 * or a text-only one because the helper model cannot see. Silence would let
 * the grounded case and the blind case read identically.
 */
export function editVisionNote(
  latestAssistant: Message | undefined,
  grounded: boolean,
): string | null {
  if (!grounded || !latestAssistant) return null;
  const context = latestAssistant.parts
    .find((part) => part.type === "generation_metadata")
    ?.metadata_json?.context as Record<string, unknown> | undefined;
  const vision = context?.vision as Record<string, unknown> | undefined;
  return vision?.visual_contents_inspected
    ? "Grounded in your source image."
    : "The helper model could not view the image, so this suggestion is text-only.";
}
