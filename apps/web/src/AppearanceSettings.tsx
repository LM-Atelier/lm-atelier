import { Monitor, Moon, Sun } from "lucide-react";
import { useId } from "react";
import { clockOptions, setClockChoice, useClockChoice, type ClockChoice } from "./clockPreference";
import {
  ROOMS,
  ROOM_LABELS,
  type Appearance,
  type ChatWidth,
  type ModeChoice,
  type MotionChoice,
  type Room,
} from "./theme";

const MODES: readonly { value: ModeChoice; label: string; Icon: typeof Sun }[] = [
  { value: "light", label: "Light", Icon: Sun },
  { value: "dark", label: "Dark", Icon: Moon },
  { value: "system", label: "System", Icon: Monitor },
];

const CHAT_WIDTH_OPTIONS: readonly { value: ChatWidth; label: string }[] = [
  { value: "standard", label: "Standard" },
  { value: "wide", label: "Wide" },
  { value: "full", label: "Full width" },
];

const MOTIONS: readonly { value: MotionChoice; label: string }[] = [
  { value: "system", label: "Automatic" },
  { value: "reduce", label: "Reduce" },
];

const CLOCKS: readonly { value: ClockChoice; label: string }[] = [
  { value: "system", label: "Automatic" },
  { value: "12", label: "12-hour" },
  { value: "24", label: "24-hour" },
];

/** A quarter to four in the afternoon, which reads differently on every clock. */
const SAMPLE_TIME = new Date(2026, 0, 1, 15, 45);

function ClockSetting() {
  const clock = useClockChoice();
  const id = useId();
  const sample = SAMPLE_TIME.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
    ...clockOptions(clock),
  });
  return (
    <section>
      <div className="detail-title"><div><h2>Times</h2><p>Saved in this browser.</p></div></div>
      <div className="setting-row appearance-row">
        <span>
          <strong id={`${id}-clock`}>Clock</strong>
          <small id={`${id}-clock-help`}>
            Automatic writes times the way your computer&apos;s language does. Times look like {sample}.
          </small>
        </span>
        <div className="segmented" role="group" aria-labelledby={`${id}-clock`} aria-describedby={`${id}-clock-help`}>
          {CLOCKS.map(({ value, label }) => (
            <button
              type="button"
              key={value}
              className={clock === value ? "active" : ""}
              aria-pressed={clock === value}
              onClick={() => setClockChoice(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

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
  const { modeChoice, setMode, room, setRoom, chatWidth, setChatWidth, motionChoice, setMotion } =
    appearance;
  const id = useId();
  return (
    <>
      <section>
        <div className="detail-title"><div><h2>Light and theme</h2><p>Saved in this browser.</p></div></div>
        <div className="setting-row appearance-row">
          <span>
            <strong id={`${id}-mode`}>Mode</strong>
            <small id={`${id}-mode-help`}>
              Every theme has a light and a dark version. System follows your computer.
            </small>
          </span>
          <div className="segmented" role="group" aria-labelledby={`${id}-mode`} aria-describedby={`${id}-mode-help`}>
            {MODES.map(({ value, label, Icon }) => (
              <button
                type="button"
                key={value}
                className={modeChoice === value ? "active" : ""}
                aria-pressed={modeChoice === value}
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
      <section>
        <div className="detail-title"><div><h2>Layout</h2><p>Saved in this browser.</p></div></div>
        <div className="setting-row appearance-row">
          <span>
            <strong id={`${id}-width`}>Chat width</strong>
            <small id={`${id}-width-help`}>
              Standard keeps lines short enough to read comfortably. Wider leaves more room for tables, code and images.
            </small>
          </span>
          <div className="segmented" role="group" aria-labelledby={`${id}-width`} aria-describedby={`${id}-width-help`}>
            {CHAT_WIDTH_OPTIONS.map(({ value, label }) => (
              <button
                type="button"
                key={value}
                className={chatWidth === value ? "active" : ""}
                aria-pressed={chatWidth === value}
                onClick={() => setChatWidth(value)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className="setting-row appearance-row">
          <span>
            <strong id={`${id}-motion`}>Motion</strong>
            <small id={`${id}-motion-help`}>
              Automatic follows your computer&apos;s setting. Reduce stops animations and smooth scrolling here.
            </small>
          </span>
          <div className="segmented" role="group" aria-labelledby={`${id}-motion`} aria-describedby={`${id}-motion-help`}>
            {MOTIONS.map(({ value, label }) => (
              <button
                type="button"
                key={value}
                className={motionChoice === value ? "active" : ""}
                aria-pressed={motionChoice === value}
                onClick={() => setMotion(value)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </section>
      <ClockSetting />
    </>
  );
}
