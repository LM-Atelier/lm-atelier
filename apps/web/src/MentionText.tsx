import type { ReactNode } from "react";
import type { MessageReference } from "./types";

/** Message text with its recorded mentions marked.
 *
 * Marks come from the record, never from the text. Scanning for `@word` would
 * be one regex and would be wrong: a message can contain `@ada-lovelace` that
 * nobody chose - typed by hand, pasted, or left behind after the reference was
 * dropped from the draft - and marking it would claim a binding that does not
 * exist. The composer refuses to bind text; so does this.
 *
 * Nothing here links anywhere. The snapshot cannot say whether the subject
 * still exists, and a link that sometimes leads nowhere is worse than no link:
 * it invites a click that fails rather than showing plainly that this is a
 * record of what a past turn used.
 */
export function MentionText({
  text,
  references = [],
}: {
  text: string;
  references?: MessageReference[];
}) {
  const marks = locate(text, references);
  if (marks.length === 0) return <div className="message-text">{text}</div>;

  const pieces: ReactNode[] = [];
  let cursor = 0;
  marks.forEach((mark, index) => {
    if (mark.at > cursor) pieces.push(text.slice(cursor, mark.at));
    pieces.push(
      <span
        key={`${mark.reference.reference_subject_id}-${index}`}
        className="message-mention"
        // The name as it was when the turn was sent, which is the whole point
        // of holding a snapshot: a rename must not rewrite an old message.
        title={mark.reference.subject_name}
      >
        {text.slice(mark.at, mark.at + mark.length)}
      </span>,
    );
    cursor = mark.at + mark.length;
  });
  if (cursor < text.length) pieces.push(text.slice(cursor));

  return <div className="message-text">{pieces}</div>;
}

type Mark = { at: number; length: number; reference: MessageReference };

/** Where each recorded mention is written, at most once per reference.
 *
 * One occurrence each, because a subject can be chosen once and then typed
 * again by hand: the text has two, exactly one is a reference, and marking
 * both would assert a binding that was never made. Marking the first is
 * arbitrary between the two, but it is the only option that does not claim
 * something untrue.
 */
function locate(text: string, references: MessageReference[]): Mark[] {
  const taken: Mark[] = [];
  for (const reference of references) {
    const written = `@${reference.mention_slug}`;
    let from = 0;
    for (;;) {
      const at = text.indexOf(written, from);
      if (at === -1) break;
      const before = at > 0 ? text[at - 1] : " ";
      const after = text[at + written.length];
      const bounded = /\s/.test(before) && (after === undefined || !/[a-z0-9-]/.test(after));
      const overlaps = taken.some((mark) => at < mark.at + mark.length && mark.at < at + written.length);
      if (bounded && !overlaps) {
        taken.push({ at, length: written.length, reference });
        break;
      }
      from = at + 1;
    }
  }
  return taken.sort((left, right) => left.at - right.at);
}
