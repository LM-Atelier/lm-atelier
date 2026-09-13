import { useId, useState } from "react";
import {
  setNotifyWhenFinished,
  useNotifyWhenFinished,
  type NotificationOutcome,
} from "./completionNotifications";
import { setSoundWhenFinished, useSoundWhenFinished } from "./completionSound";
import { setSendKeyChoice, useSendKeyChoice, type SendKeyChoice } from "./sendKey";

const SEND_KEYS: readonly { value: SendKeyChoice; label: string }[] = [
  { value: "enter", label: "Enter sends" },
  { value: "mod-enter", label: "Ctrl+Enter sends" },
];

const REFUSED_TEXT: Partial<Record<NotificationOutcome, string>> = {
  denied: "This browser has blocked notifications for the workspace. Allow them in the browser's site settings, then turn this on again.",
  unsupported: "This browser cannot show notifications.",
};

function NotificationSetting() {
  const on = useNotifyWhenFinished();
  const [refusal, setRefusal] = useState<string | null>(null);
  const id = useId();
  const choose = (next: boolean) => {
    setRefusal(null);
    void setNotifyWhenFinished(next).then((outcome) => setRefusal(REFUSED_TEXT[outcome] ?? null));
  };
  return (
    <section>
      <div className="detail-title"><div><h2>Notifications</h2><p>Saved in this browser.</p></div></div>
      <div className="setting-row appearance-row">
        <span>
          <strong id={`${id}-notify`}>When work finishes</strong>
          <small id={`${id}-notify-help`}>
            Only while the workspace is in the background, and without what was asked or answered.
          </small>
        </span>
        <div className="segmented" role="group" aria-labelledby={`${id}-notify`} aria-describedby={`${id}-notify-help`}>
          <button type="button" className={on ? "" : "active"} aria-pressed={!on} onClick={() => choose(false)}>
            Stay quiet
          </button>
          <button type="button" className={on ? "active" : ""} aria-pressed={on} onClick={() => choose(true)}>
            Notify me
          </button>
        </div>
      </div>
      {refusal && <p className="muted" role="status">{refusal}</p>}
      <SoundSetting />
    </section>
  );
}

function SoundSetting() {
  const on = useSoundWhenFinished();
  const [unavailable, setUnavailable] = useState(false);
  const id = useId();
  const choose = (next: boolean) => setUnavailable(!setSoundWhenFinished(next));
  return (
    <>
      <div className="setting-row appearance-row">
        <span>
          <strong id={`${id}-sound`}>Sound</strong>
          <small id={`${id}-sound-help`}>
            A short chime at the same moments, with or without notifications. Turning it on plays it once.
          </small>
        </span>
        <div className="segmented" role="group" aria-labelledby={`${id}-sound`} aria-describedby={`${id}-sound-help`}>
          <button type="button" className={on ? "" : "active"} aria-pressed={!on} onClick={() => choose(false)}>
            Silent
          </button>
          <button type="button" className={on ? "active" : ""} aria-pressed={on} onClick={() => choose(true)}>
            Play a chime
          </button>
        </div>
      </div>
      {unavailable && <p className="muted" role="status">This browser cannot play sounds.</p>}
    </>
  );
}

/** Everyday behaviour that is not about how the workspace looks.
 *
 * Which keystroke sends, and whether finished work is announced while you are
 * looking at something else.
 */
export function GeneralSettings() {
  const sendKey = useSendKeyChoice();
  const id = useId();
  return (
    <>
      <section>
        <div className="detail-title"><div><h2>Sending</h2><p>Saved in this browser.</p></div></div>
        <div className="setting-row appearance-row">
          <span>
            <strong id={`${id}-send`}>Send key</strong>
            <small id={`${id}-send-help`}>
              With Ctrl+Enter (Cmd+Enter on a Mac), Enter starts a new line instead. Shift+Enter always starts a new line.
            </small>
          </span>
          <div className="segmented" role="group" aria-labelledby={`${id}-send`} aria-describedby={`${id}-send-help`}>
            {SEND_KEYS.map(({ value, label }) => (
              <button
                type="button"
                key={value}
                className={sendKey === value ? "active" : ""}
                aria-pressed={sendKey === value}
                onClick={() => setSendKeyChoice(value)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </section>
      <NotificationSetting />
    </>
  );
}
