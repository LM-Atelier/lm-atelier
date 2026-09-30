import type { ChatDetail, Message, MessagePart } from "./types";

/** Which of an edit's results is being made, when it asked for more than one. */
export type StudioRunningPlace = { index: number; count: number };

/** Where the newest edit still running in a studio session has got to.
 *
 * The server keeps a progress part on each answer while its work runs, with
 * the step it is on and how far along it is, and the session is read again
 * every few seconds while anything is pending. An edit that asked for several
 * results has an answer for each, made one after another, so what it shows is
 * the first of them not yet made, and its place among them. Null when nothing
 * is running.
 */
export function studioApplyProgress(
  session: ChatDetail | null | undefined,
): { part: MessagePart; place: StudioRunningPlace | null } | null {
  if (!session) return null;
  const messages = session.messages;
  let newest = messages.length - 1;
  while (newest >= 0 && !(messages[newest].role === "assistant" && messages[newest].status === "pending")) newest -= 1;
  if (newest < 0) return null;
  const answers = answersBeside(messages, newest);
  const running = answers.find((message) => message.status === "pending") ?? messages[newest];
  const part = running.parts.find((item) => item.type === "progress");
  if (!part) return null;
  return {
    part,
    place: answers.length > 1 ? { index: answers.indexOf(running) + 1, count: answers.length } : null,
  };
}

/** Why the newest edit came back without a picture, when any of its results did.
 *
 * A failed answer carries no picture, so it never joins the strip: the running
 * edit's progress simply went away, after its words had already been cleared,
 * and nothing said that it had failed. The server replaces a failed answer's
 * parts with the reason. Only the newest request is read, since the next edit
 * takes its place; a request with no answer yet has nothing to say. The
 * request's id comes with the words, so each failure is said afresh.
 */
export function studioApplyFailure(
  session: ChatDetail | null | undefined,
): { requestId: string; text: string } | null {
  if (!session) return null;
  const messages = session.messages;
  let request = messages.length - 1;
  while (request >= 0 && messages[request].role !== "user") request -= 1;
  if (request < 0) return null;
  const answers = messages.slice(request + 1).filter((message) => message.role === "assistant");
  const failed = answers.filter((message) => message.status === "failed");
  if (failed.length === 0) return null;
  const reason = failed[0].parts.find((part) => part.type === "error" && part.text?.trim())?.text?.trim();
  const what = answers.length > 1
    ? `${failed.length} of ${answers.length} results did not finish`
    : "The edit did not finish";
  return { requestId: messages[request].id, text: reason ? `${what}: ${reason}` : `${what}.` };
}

/** Every answer to the request the answer at `at` belongs to: they sit between it and the next one. */
function answersBeside(messages: Message[], at: number): Message[] {
  let start = at;
  while (start > 0 && messages[start - 1].role !== "user") start -= 1;
  let end = at + 1;
  while (end < messages.length && messages[end].role !== "user") end += 1;
  return messages.slice(start, end).filter((message) => message.role === "assistant");
}
