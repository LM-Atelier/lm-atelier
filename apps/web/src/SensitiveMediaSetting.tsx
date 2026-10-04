import { useId, useState } from "react";
import {
  SENSITIVE_MEDIA_CHOICES,
  setSensitiveMediaChoice,
  useSensitiveMediaChoice,
  type SensitiveMediaChoice,
} from "./sensitiveMedia";

const LABELS: Record<SensitiveMediaChoice, string> = { show: "Show", blur: "Blur", hide: "Hide" };

/** Whether pictures and videos in chats appear at once, blurred, or hidden until shown. */
export function SensitiveMediaSetting() {
  const choice = useSensitiveMediaChoice();
  const [unsaved, setUnsaved] = useState(false);
  const id = useId();
  return (
    <section>
      <div className="detail-title"><div><h2>Privacy on screen</h2><p>Saved in this browser.</p></div></div>
      <div className="setting-row appearance-row">
        <span>
          <strong id={`${id}-media`}>Pictures and videos in chats</strong>
          <small id={`${id}-media-help`}>
            Blurred or hidden ones stay covered until you choose Show on one, and are covered again when you leave. This keeps them off your screen; it does not encrypt anything.
          </small>
        </span>
        <div className="segmented" role="group" aria-labelledby={`${id}-media`} aria-describedby={`${id}-media-help`}>
          {SENSITIVE_MEDIA_CHOICES.map((value) => (
            <button
              type="button"
              key={value}
              className={choice === value ? "active" : ""}
              aria-pressed={choice === value}
              onClick={() => setUnsaved(!setSensitiveMediaChoice(value))}
            >
              {LABELS[value]}
            </button>
          ))}
        </div>
      </div>
      {unsaved && <p className="muted" role="status">This browser cannot save the choice, so it was not changed.</p>}
    </section>
  );
}
