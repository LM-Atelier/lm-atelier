import { useId } from "react";
import { SETTING_DETAILS, setSettingDetail, useSettingDetail } from "./settingDetail";
import type { Visibility } from "./settings";

const LABELS: Record<Visibility, string> = {
  basic: "Basic",
  advanced: "Advanced",
  expert: "Expert",
};

/** Where profile, preset and chat settings editors start, as a Models & generation setting. */
export function SettingDetailSetting() {
  const level = useSettingDetail();
  const id = useId();
  return (
    <section>
      <div className="detail-title"><div><h2>Setting detail</h2><p>Saved in this browser.</p></div></div>
      <div className="setting-row appearance-row">
        <span>
          <strong id={`${id}-detail`}>Editors open at</strong>
          <small id={`${id}-detail-help`}>
            How many settings a profile, preset or chat settings editor shows when it opens. You can still switch inside an editor.
          </small>
        </span>
        <div className="segmented" role="group" aria-labelledby={`${id}-detail`} aria-describedby={`${id}-detail-help`}>
          {SETTING_DETAILS.map((value) => (
            <button
              type="button"
              key={value}
              className={level === value ? "active" : ""}
              aria-pressed={level === value}
              onClick={() => setSettingDetail(value)}
            >
              {LABELS[value]}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
