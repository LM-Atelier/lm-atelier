import { useState } from "react";
import { survivingMentions, turnReferences, type TrackedMention } from "./mentionDraft";

/** What the composer chose, and what it will send.
 *
 * Kept beside the draft rather than derived from it. The reconciliation runs
 * at send rather than on every keystroke, because text reaches the composer
 * from several places - quoting a message, animating an image, accepting the
 * prompt helper's rewrite - and a rule applied at only some of them would be
 * worse than no rule at all.
 *
 * The reconciliation only ever narrows: deleting a mention from the text drops
 * its reference, and typing one can never add it.
 */
export function useComposerMentions() {
  const [chosen, setChosen] = useState<TrackedMention[]>([]);
  return {
    add: (mention: TrackedMention) => setChosen((current) => [...current, mention]),
    clear: () => setChosen([]),
    /** The references still written in this text, in the server's shape. */
    forText: (text: string) => turnReferences(survivingMentions(text, chosen)),
  };
}
