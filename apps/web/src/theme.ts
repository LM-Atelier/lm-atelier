import { useCallback, useEffect, useMemo, useState } from "react";

/** Which room you are working in, and whether its light is on.
 *
 * Two independent axes. The **room** is the theme - one complete, cohesive
 * palette - and the **mode** is whether that room is lit or dark. Every room
 * supplies both, so changing one never changes the other.
 *
 * An earlier design made the room a property of the view: prose on paper,
 * the studio in the dark, on the theory that colour cannot be judged against
 * a warm ground. It read as one interface disagreeing with itself, because a
 * dark sidebar beside a paper chat is not "two rooms" to anyone who did not
 * design it. Changing room is now something a person does, not something a
 * screen does to them.
 */
export type ThemeMode = "light" | "dark";

/** What a person chose for the light: one mode, or whatever the computer is set to. */
export type ModeChoice = ThemeMode | "system";

/** A room is a whole palette. Adding one means adding a block of custom
 * properties and a name here; no rule in the stylesheet changes. */
export const ROOMS = ["north-light", "blue-hour"] as const;
export type Room = (typeof ROOMS)[number];

export const ROOM_LABELS: Record<Room, string> = {
  "north-light": "North Light",
  "blue-hour": "Blue Hour",
};

export const ROOM_KEY = "local-lm-room";
export const MODE_KEY = "local-lm-mode";

export function isRoom(value: unknown): value is Room {
  return typeof value === "string" && (ROOMS as readonly string[]).includes(value);
}

export function isMode(value: unknown): value is ThemeMode {
  return value === "light" || value === "dark";
}

export function isModeChoice(value: unknown): value is ModeChoice {
  return isMode(value) || value === "system";
}

const PREFERS_LIGHT = "(prefers-color-scheme: light)";

/** The mode the operating system is asking for right now, dark when it cannot say. */
function systemMode(): ThemeMode {
  return typeof matchMedia === "function" && matchMedia(PREFERS_LIGHT).matches ? "light" : "dark";
}

export function storedRoom(): Room {
  const stored = localStorage.getItem(ROOM_KEY);
  return isRoom(stored) ? stored : "north-light";
}

/** The remembered choice, or a mode seeded once from the system when there is none.
 *
 * Without a choice the system preference seeds the first answer and it then
 * stays put, because an interface that flips itself at sunset without being
 * asked has changed without being asked. Following the system is still
 * available - as a choice somebody makes, "system", rather than a default
 * nobody did.
 */
export function storedModeChoice(): ModeChoice {
  const stored = localStorage.getItem(MODE_KEY);
  return isModeChoice(stored) ? stored : systemMode();
}

export function useRoom(): [Room, (room: Room) => void] {
  const [room, setRoom] = useState<Room>(storedRoom);
  const choose = useCallback((next: Room) => {
    setRoom(next);
    localStorage.setItem(ROOM_KEY, next);
  }, []);
  return [room, choose];
}

/** The mode choice, the mode it currently means, and a way to change the choice.
 *
 * "system" is the one choice whose meaning changes on its own, so only while it
 * is chosen does this listen for the operating system switching between light
 * and dark.
 */
export function useThemeMode(): [ModeChoice, ThemeMode, (choice: ModeChoice) => void] {
  const [choice, setChoice] = useState<ModeChoice>(storedModeChoice);
  const [system, setSystem] = useState<ThemeMode>(systemMode);
  useEffect(() => {
    if (choice !== "system" || typeof matchMedia !== "function") return;
    const query = matchMedia(PREFERS_LIGHT);
    const follow = () => setSystem(query.matches ? "light" : "dark");
    follow();
    query.addEventListener("change", follow);
    return () => query.removeEventListener("change", follow);
  }, [choice]);
  const choose = useCallback((next: ModeChoice) => {
    setChoice(next);
    localStorage.setItem(MODE_KEY, next);
  }, []);
  return [choice, choice === "system" ? system : choice, choose];
}

/** The room and its light, remembered and applied to the document.
 *
 * Both attributes go on the document element rather than on a wrapper, so a
 * dialog rendered through a portal is in the same room as everything else.
 *
 * The returned object keeps its identity until the room or the light changes.
 * Settings renders inside a memoized view, and an object rebuilt on every
 * render would rebuild that view on every render too.
 */
export interface Appearance {
  room: Room;
  /** The mode in effect: what the document is drawn in. */
  mode: ThemeMode;
  /** What was chosen, which may be to follow the system. */
  modeChoice: ModeChoice;
  setRoom: (room: Room) => void;
  setMode: (choice: ModeChoice) => void;
}

export function useAppearance(): Appearance {
  const [room, setRoom] = useRoom();
  const [modeChoice, mode, setMode] = useThemeMode();
  useEffect(() => {
    document.documentElement.dataset.room = room;
    document.documentElement.dataset.mode = mode;
  }, [room, mode]);
  return useMemo(
    () => ({ room, mode, modeChoice, setRoom, setMode }),
    [room, mode, modeChoice, setRoom, setMode],
  );
}
