import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import { api } from "./api";
import { useAppearance } from "./theme";
import { applyWorkspaceLockStatus, workspaceLockFailure } from "./workspaceLockState";

const REFUSED = "LM Atelier could not be unlocked.";
const SETTING_UNREADABLE = "The saved lock setting could not be read, so LM Atelier stays locked. The privacy documentation explains how to reset it.";

function waitText(seconds: number): string {
  return `Try again in ${seconds} ${seconds === 1 ? "second" : "seconds"}.`;
}

/** What stands in for the workspace while it is locked.
 *
 * A page rather than a dialog: there is nothing behind it to return to, and
 * nothing to dismiss. Every refusal reads the same, so the page never says
 * whether a PIN was close, and never how many tries are left. The PIN is held
 * only in the field, which is emptied as soon as it is sent.
 */
export function LockedWorkspace({ requirePin }: { requirePin: boolean }) {
  // The workspace is unmounted, so nothing else keeps the chosen theme on the page.
  useAppearance();
  const id = useId();
  const pinField = useRef<HTMLInputElement>(null);
  const unlockButton = useRef<HTMLButtonElement>(null);
  const sending = useRef(false);
  const [pin, setPin] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [waitSeconds, setWaitSeconds] = useState(0);
  const [announcement, setAnnouncement] = useState("");

  useEffect(() => {
    (requirePin ? pinField.current : unlockButton.current)?.focus();
  }, [requirePin]);

  // Filled a moment after the page appears, so screen readers hear it as news
  // rather than as part of a page that was already there.
  useEffect(() => {
    const timer = window.setTimeout(() => setAnnouncement("LM Atelier is locked."), 100);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (waitSeconds <= 0) return;
    const timer = window.setTimeout(() => {
      setWaitSeconds((seconds) => Math.max(0, seconds - 1));
      if (waitSeconds === 1) setAnnouncement("You can try again now.");
    }, 1_000);
    return () => window.clearTimeout(timer);
  }, [waitSeconds]);

  const unavailable = busy || waitSeconds > 0;
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (sending.current || unavailable) return;
    const attempt = pin;
    setPin("");
    if (requirePin && !attempt) {
      setError("Enter the PIN.");
      pinField.current?.focus();
      return;
    }
    sending.current = true;
    setBusy(true);
    setError("");
    try {
      // Taking the answer brings the workspace back; this page goes with it.
      applyWorkspaceLockStatus(await api.unlockWorkspace(requirePin ? attempt : undefined));
    } catch (failure) {
      const refusal = workspaceLockFailure(failure);
      setError(refusal.code === "workspace-lock-setting-invalid" ? SETTING_UNREADABLE : REFUSED);
      if (refusal.retryAfterSeconds > 0) {
        setWaitSeconds(refusal.retryAfterSeconds);
        setAnnouncement(`Unlocking is paused. ${waitText(refusal.retryAfterSeconds)}`);
      }
    } finally {
      sending.current = false;
      setBusy(false);
      (requirePin ? pinField.current : unlockButton.current)?.focus();
    }
  };

  const described = [error ? `${id}-error` : "", waitSeconds > 0 ? `${id}-wait` : ""].filter(Boolean).join(" ") || undefined;
  return (
    <main id="main-content" tabIndex={-1} className="first-run-shell workspace-lock-shell">
      <div className="workspace-lock-card">
        <h1>LM Atelier is locked</h1>
        <p>Work that was already running keeps going while the workspace is locked.</p>
        <form className="workspace-lock-form" onSubmit={(event) => void submit(event)}>
          {requirePin && (
            <label htmlFor={`${id}-pin`}>
              PIN
              <input
                id={`${id}-pin`}
                ref={pinField}
                type="password"
                autoComplete="off"
                value={pin}
                aria-invalid={error ? true : undefined}
                aria-describedby={described}
                onChange={(event) => setPin(event.target.value)}
              />
            </label>
          )}
          <div className="workspace-lock-actions">
            <button
              ref={unlockButton}
              type="submit"
              className="primary"
              aria-disabled={unavailable}
              aria-describedby={requirePin ? undefined : described}
            >
              {busy ? "Unlocking…" : "Unlock"}
            </button>
          </div>
        </form>
        {error && <p id={`${id}-error`} className="callout error" role="alert">{error}</p>}
        {waitSeconds > 0 && <p id={`${id}-wait`}>{waitText(waitSeconds)}</p>}
        <p className="sr-only" role="status">{announcement}</p>
      </div>
    </main>
  );
}
