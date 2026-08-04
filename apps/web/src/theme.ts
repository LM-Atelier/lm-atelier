import { useState } from "react";

/** Which light the workspace is in, and who decides it.
 *
 * The two rooms - paper for reading, dark for making - are a property of
 * what a surface is for. This is a different question: what the person at
 * the keyboard prefers, which overrides the rooms entirely when they say
 * so. Someone working at night wants the whole thing dark no matter how
 * good paper is for prose, and that is not a preference to argue with.
 *
 * "By room" is the default because it is the design's own answer. The two
 * fixed choices exist because a default is not the same as a decision.
 */
export type ThemeChoice = "by-room" | "light" | "dark";

export const THEME_KEY = "local-lm-theme";

export function isThemeChoice(value: unknown): value is ThemeChoice {
  return value === "by-room" || value === "light" || value === "dark";
}

export function storedTheme(): ThemeChoice {
  const stored = localStorage.getItem(THEME_KEY);
  return isThemeChoice(stored) ? stored : "by-room";
}

/** The room a surface should render in, once the person has had their say.
 *
 * A fixed choice wins over the surface's own nature: picking "light" means
 * the studio is on paper too, which is worse for judging colour and is
 * still what was asked for.
 */
export function roomFor(choice: ThemeChoice, prefersReading: boolean): "reading" | "making" {
  if (choice === "light") return "reading";
  if (choice === "dark") return "making";
  return prefersReading ? "reading" : "making";
}

/** The appearance choice, remembered across sessions.
 *
 * Kept here rather than in the component so the storage key and the
 * fallback live with the type that defines them.
 */
export function useThemeChoice(): [ThemeChoice, (choice: ThemeChoice) => void] {
  const [choice, setChoice] = useState<ThemeChoice>(storedTheme);
  return [
    choice,
    (next) => {
      setChoice(next);
      localStorage.setItem(THEME_KEY, next);
    },
  ];
}
