import { Moon, Sun } from "lucide-react";
import { useId } from "react";
import { ROOMS, ROOM_LABELS, type Appearance, type Room, type ThemeMode } from "./theme";

const MODES: readonly { value: ThemeMode; label: string; Icon: typeof Sun }[] = [
  { value: "light", label: "Light", Icon: Sun },
  { value: "dark", label: "Dark", Icon: Moon },
];

/** Whether the light is on, and which room you are in, as a Settings page.
 *
 * These were a pill floating over the bottom-right corner of every screen, on
 * top of whatever was underneath, which put a rarely changed choice in front of
 * the work all day. Here they are two ordinary settings beside the others.
 *
 * Both values still live in the workspace's one appearance state and are
 * remembered under the same keys, so moving the control resets nobody's choice
 * and there is no second copy to disagree with the first.
 *
 * The theme choice is hidden while there is only one room. A picker with a
 * single entry asks a question that has no second answer.
 */
export function AppearanceSettings({ appearance }: { appearance: Appearance }) {
  const { mode, setMode, room, setRoom } = appearance;
  const id = useId();
  return (
    <section>
      <div className="detail-title"><div><h2>Light and theme</h2><p>Saved in this browser.</p></div></div>
      <div className="setting-row appearance-row">
        <span>
          <strong id={`${id}-mode`}>Mode</strong>
          <small id={`${id}-mode-help`}>Every theme has a light and a dark version.</small>
        </span>
        <div className="segmented" role="group" aria-labelledby={`${id}-mode`} aria-describedby={`${id}-mode-help`}>
          {MODES.map(({ value, label, Icon }) => (
            <button
              type="button"
              key={value}
              className={mode === value ? "active" : ""}
              aria-pressed={mode === value}
              onClick={() => setMode(value)}
            >
              <Icon size={15} aria-hidden="true" />{label}
            </button>
          ))}
        </div>
      </div>
      {ROOMS.length > 1 && (
        <div className="setting-row appearance-row">
          <span>
            <strong id={`${id}-theme`}>Theme</strong>
            <small id={`${id}-theme-help`}>A whole palette, changed everywhere at once.</small>
          </span>
          <select
            aria-labelledby={`${id}-theme`}
            aria-describedby={`${id}-theme-help`}
            value={room}
            onChange={(event) => setRoom(event.target.value as Room)}
          >
            {ROOMS.map((value) => (
              <option key={value} value={value}>
                {ROOM_LABELS[value]}
              </option>
            ))}
          </select>
        </div>
      )}
    </section>
  );
}
