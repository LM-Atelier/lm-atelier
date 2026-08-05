import { Moon, Sun } from "lucide-react";
import { ROOMS, ROOM_LABELS, type Appearance, type Room, type ThemeMode } from "./theme";

/** Turn the light on or off, and choose which room you are in.
 *
 * Two controls because they are two questions. The mode is the one people
 * change often and expect to find; the room is a whole palette and changes
 * rarely, so it sits beside the switch rather than competing with it.
 *
 * The room control is hidden while there is only one room. A picker with a
 * single entry asks a question that has no second answer.
 */
export function ThemeToggle({ appearance }: { appearance: Appearance }) {
  const { mode, setMode: onMode, room, setRoom: onRoom } = appearance;
  const next: ThemeMode = mode === "dark" ? "light" : "dark";
  const label = mode === "dark" ? "Switch to light" : "Switch to dark";
  const Icon = mode === "dark" ? Sun : Moon;
  return (
    <div className="theme-toggle" role="group" aria-label="Appearance">
      <button
        type="button"
        className="icon-button"
        aria-label={label}
        title={label}
        onClick={() => onMode(next)}
      >
        <Icon size={15} aria-hidden="true" />
      </button>
      {ROOMS.length > 1 && (
        <select
          aria-label="Theme"
          value={room}
          onChange={(event) => onRoom(event.target.value as Room)}
        >
          {ROOMS.map((value) => (
            <option key={value} value={value}>
              {ROOM_LABELS[value]}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
