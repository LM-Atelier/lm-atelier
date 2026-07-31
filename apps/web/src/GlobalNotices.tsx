import { X } from "lucide-react";

/** The two things the app has to say for itself: a failure, and a dead socket. */

export interface FailingMutation {
  error: Error | null;
}

/** The first failure among these mutations, so a chain cannot omit one silently. */
function firstError(mutations: FailingMutation[]): Error | null {
  return mutations.find((mutation) => mutation.error)?.error ?? null;
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
  const failure = firstError(mutations);
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
          <X size={16} />
          {failure.message}
        </div>
      )}
    </>
  );
}
