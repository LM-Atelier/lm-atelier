import type { ChatDetail, MessagePart } from "./types";

/** Where the newest edit still running in a studio session has got to.
 *
 * The server keeps a progress part on each answer while its work runs, with
 * the step it is on and how far along it is, and the session is read again
 * every few seconds while anything is pending. Null when nothing is running.
 */
export function studioApplyProgress(session: ChatDetail | null | undefined): MessagePart | null {
  if (!session) return null;
  for (let index = session.messages.length - 1; index >= 0; index -= 1) {
    const message = session.messages[index];
    if (message.role !== "assistant" || message.status !== "pending") continue;
    return message.parts.find((part) => part.type === "progress") ?? null;
  }
  return null;
}
