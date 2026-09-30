import { ErrorCallout } from "./ErrorCallout";
import { studioApplyFailure } from "./studioApplyProgress";
import type { ChatDetail } from "./types";

/** Says so, and why, when the newest edit came back without a picture.
 *
 * Keyed by the request, so dismissing one failure never hides the next one's,
 * even when both give the same reason.
 */
export function StudioApplyFailure({ session }: { session: ChatDetail | null }) {
  const failure = studioApplyFailure(session);
  return failure ? <ErrorCallout key={failure.requestId} message={failure.text} /> : null;
}
