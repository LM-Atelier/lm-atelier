import { Moon, Sun, SunMoon } from "lucide-react";
import type { ThemeChoice } from "./theme";

const CHOICES: Array<{ value: ThemeChoice; label: string; icon: typeof Sun }> = [
  { value: "by-room", label: "Light that suits each room", icon: SunMoon },
  { value: "light", label: "Always light", icon: Sun },
  { value: "dark", label: "Always dark", icon: Moon },
];

/** Pick the light to work in.
 *
 * The middle option is the design's own answer - paper where you read,
 * a dark room where you judge pictures - and the other two exist because a
 * default is not a decision. Someone working at night wants the whole thing
 * dark whatever prose prefers, and that is not an argument to have.
 */
export function ThemeToggle({
  choice,
  onChoose,
}: {
  choice: ThemeChoice;
  onChoose: (choice: ThemeChoice) => void;
}) {
  return (
    <div className="theme-toggle" role="group" aria-label="Appearance">
      {CHOICES.map(({ value, label, icon: Icon }) => (
        <button
          key={value}
          type="button"
          className={`icon-button ${choice === value ? "selected" : ""}`}
          aria-label={label}
          aria-pressed={choice === value}
          title={label}
          onClick={() => onChoose(value)}
        >
          <Icon size={15} aria-hidden="true" />
        </button>
      ))}
    </div>
  );
}
