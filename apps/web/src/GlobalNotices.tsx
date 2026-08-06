import { useState } from "react";
import { X } from "lucide-react";

/** The two things the app has to say for itself: a failure, and a dead socket. */

export interface FailingMutation {
  error: Error | null;
}

/** The first failure among these mutations that has not been read yet.
 *
 * Dismissal has to be part of this choice rather than a test applied after
 * it. Picking the first failure and then hiding it if dismissed means a
 * mutation that failed once, was read, and still holds its error shadows
 * every later failure behind it - which is the silent failure this component
 * exists to prevent, reintroduced by the dismiss button.
 */
function firstUnreadError(
  mutations: FailingMutation[],
  dismissed: ReadonlySet<Error>,
): Error | null {
  return mutations.find((mutation) => mutation.error && !dismissed.has(mutation.error))?.error
    ?? null;
}

export function GlobalNotices({
  connected,
  mutations,
}: {
  connected: boolean;
  mutations: FailingMutation[];
}) {
  // Read once from one list. The two hand-written `||` chains this replaces had
  // to be kept identical by hand and had drifted: both omitted exportProject, so
  // a failed project export produced no message at all.
  // Dismissal is by error *instance*, not message text: a retried action that
  // fails again produces a new Error and the toast rightly returns, while the
  // one the user already read stays dismissed. A set rather than one slot,
  // because reading a second failure must not un-dismiss the first.
  const [dismissed, setDismissed] = useState<ReadonlySet<Error>>(() => new Set());
  const failure = firstUnreadError(mutations, dismissed);
  return (
    <>
      {!connected && (
        <div className="toast" role="status">
          <span className="status-dot offline" aria-hidden />
          Live updates are disconnected. Reconnecting - what you see may be out of date.
        </div>
      )}
      {failure && (
        <div className="toast error" role="alert">
          {failure.message}
          <button
            type="button"
            className="callout-dismiss"
            aria-label="Dismiss error"
            onClick={() => setDismissed((read) => new Set(read).add(failure))}
          >
            <X size={16} />
          </button>
        </div>
      )}
    </>
  );
}
