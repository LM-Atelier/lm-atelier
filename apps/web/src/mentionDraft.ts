/** Composer state for @-mentions that become structured references.
 *
 * One rule drives all of it: **a reference is created by inserting it, never
 * by reading the text.** The server refuses to recover references by scanning
 * a prompt for `@name`, because that path "silently binds whoever the text
 * most resembles" - so the composer must not do it either. Typing the
 * characters `@ada` by hand attaches nothing. Choosing Ada from the picker
 * attaches her id, and that id is what travels.
 *
 * Editing the text can therefore only ever *remove* a reference, never add or
 * change one. That asymmetry is the whole design.
 */

/** A mention the person actually chose, and the subject it stands for. */
export type TrackedMention = {
  /** The subject id. This came from the picker; it is never parsed back out. */
  referenceSubjectId: string;
  /** The addressing token as written, used only to notice deletion. */
  mentionSlug: string;
};

/** What a turn sends. Deliberately the server's shape, not a private one. */
export type TurnReference = {
  reference_subject_id: string;
  source: "mention";
};

/** Mentions run to the first character that cannot be in a slug. */
const SLUG_CHARACTER = /[a-z0-9-]/;

/** The partial mention being typed at the caret, without its `@`.
 *
 * Returns null when the caret is not inside one, so the picker opens only
 * while a mention is actually being written. An empty string is a real answer
 * - `@` alone means "show me everything" - which is why this is not just a
 * truthiness check at the call site.
 */
export function mentionQuery(text: string, caret: number): string | null {
  const before = text.slice(0, caret);
  const at = before.lastIndexOf("@");
  if (at === -1) return null;
  const typed = before.slice(at + 1);
  // A mention starts a word. "email@example" is an address, not a mention.
  const preceding = at > 0 ? before[at - 1] : " ";
  if (!/\s/.test(preceding)) return null;
  if (typed && !typed.split("").every((character) => SLUG_CHARACTER.test(character))) {
    return null;
  }
  return typed;
}

/** Replace the partial mention at the caret with a chosen one. */
export function insertMention(
  text: string,
  caret: number,
  mentionSlug: string,
): { text: string; caret: number } {
  // Replace only a mention that is actually being typed *here*. Searching back
  // for the nearest "@" unconditionally found the previous mention's, so
  // inserting a second one overwrote the first.
  const active = mentionQuery(text, caret);
  const before = text.slice(0, caret);
  const start = active === null ? caret : before.lastIndexOf("@");
  // A trailing space so the next word is not swallowed into the mention, and
  // so a second mention can be typed straight afterwards.
  const written = `@${mentionSlug} `;
  const next = text.slice(0, start) + written + text.slice(caret);
  return { text: next, caret: start + written.length };
}

/** Whether a written mention still stands in the text.
 *
 * Bounded on both sides so `@ada` does not count as still present when the
 * text now reads `@ada-lovelace`: those address different subjects, and
 * treating one as the other is the silent misbinding this whole design
 * refuses.
 */
function stillWritten(text: string, mentionSlug: string): boolean {
  let from = 0;
  for (;;) {
    const at = text.indexOf(`@${mentionSlug}`, from);
    if (at === -1) return false;
    const after = text[at + mentionSlug.length + 1];
    const before = at > 0 ? text[at - 1] : " ";
    if (/\s/.test(before) && (after === undefined || !SLUG_CHARACTER.test(after))) return true;
    from = at + 1;
  }
}

/** The mentions that survive an edit.
 *
 * Only ever shrinks. Anything the person deleted from the text stops being a
 * reference; nothing here can promote text into one.
 */
export function survivingMentions(text: string, tracked: TrackedMention[]): TrackedMention[] {
  const seen = new Set<string>();
  return tracked.filter((mention) => {
    if (seen.has(mention.referenceSubjectId)) return false;
    if (!stillWritten(text, mention.mentionSlug)) return false;
    seen.add(mention.referenceSubjectId);
    return true;
  });
}

/** What the turn carries. Empty when nothing was chosen. */
export function turnReferences(tracked: TrackedMention[]): TurnReference[] {
  return tracked.map((mention) => ({
    reference_subject_id: mention.referenceSubjectId,
    source: "mention" as const,
  }));
}
