import { useEffect, useState } from "react";

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

export function storedRoom(): Room {
  const stored = localStorage.getItem(ROOM_KEY);
  return isRoom(stored) ? stored : "north-light";
}

/** Dark unless asked otherwise, and asked once rather than guessed each time.
 *
 * The system preference seeds the first answer; after that the choice is the
 * person's, because an interface that flips itself at sunset is one that
 * changed without being asked.
 */
export function storedMode(): ThemeMode {
  const stored = localStorage.getItem(MODE_KEY);
  if (isMode(stored)) return stored;
  const prefersLight =
    typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: light)").matches;
  return prefersLight ? "light" : "dark";
}

export function useRoom(): [Room, (room: Room) => void] {
  const [room, setRoom] = useState<Room>(storedRoom);
  return [
    room,
    (next) => {
      setRoom(next);
      localStorage.setItem(ROOM_KEY, next);
    },
  ];
}

export function useThemeMode(): [ThemeMode, (mode: ThemeMode) => void] {
  const [mode, setMode] = useState<ThemeMode>(storedMode);
  return [
    mode,
    (next) => {
      setMode(next);
      localStorage.setItem(MODE_KEY, next);
    },
  ];
}

/** The room and its light, remembered and applied to the document.
 *
 * Both attributes go on the document element rather than on a wrapper, so a
 * dialog rendered through a portal is in the same room as everything else.
 */
export interface Appearance {
  room: Room;
  mode: ThemeMode;
  setRoom: (room: Room) => void;
  setMode: (mode: ThemeMode) => void;
}

export function useAppearance(): Appearance {
  const [room, setRoom] = useRoom();
  const [mode, setMode] = useThemeMode();
  useEffect(() => {
    document.documentElement.dataset.room = room;
    document.documentElement.dataset.mode = mode;
  }, [room, mode]);
  return { room, mode, setRoom, setMode };
}
