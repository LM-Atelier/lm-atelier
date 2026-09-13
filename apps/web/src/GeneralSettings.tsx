import { useId } from "react";
import { setSendKeyChoice, useSendKeyChoice, type SendKeyChoice } from "./sendKey";

const SEND_KEYS: readonly { value: SendKeyChoice; label: string }[] = [
  { value: "enter", label: "Enter sends" },
  { value: "mod-enter", label: "Ctrl+Enter sends" },
];

/** Everyday behaviour that is not about how the workspace looks.
 *
 * It starts with the one question every text field that sends has to answer:
 * which keystroke means "send" and which means "new line".
 */
export function GeneralSettings() {
  const sendKey = useSendKeyChoice();
  const id = useId();
  return (
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
  );
}
