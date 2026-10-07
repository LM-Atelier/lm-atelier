import { ErrorCallout } from "./ErrorCallout";
import { studioApplyFailure } from "./studioApplyProgress";
import type { ChatDetail } from "./types";

/** The words an edit was sent with, and the request that carried them. */
export type StudioSentWords = { requestId: string; words: string };

/** Says so, and why, when the newest edit came back without a picture.
 *
 * Keyed by the request, so dismissing one failure never hides the next one's,
 * even when both give the same reason. The words typed for the edit were
 * cleared once it was taken, so when they are known they are offered back,
 * ready to send again or to change first.
 */
export function StudioApplyFailure({
  session,
  sent = null,
  onWordsBack,
}: {
  session: ChatDetail | null;
  sent?: StudioSentWords | null;
  onWordsBack?: (words: string) => void;
}) {
  const failure = studioApplyFailure(session);
  if (!failure) return null;
  const words = sent && sent.requestId === failure.requestId && sent.words.trim() ? sent.words : null;
  return (
    <ErrorCallout
      key={failure.requestId}
      message={failure.text}
      action={words && onWordsBack ? (
        <button type="button" className="secondary compact-button" onClick={() => onWordsBack(words)}>
          Use the words again
        </button>
      ) : undefined}
    />
  );
}
