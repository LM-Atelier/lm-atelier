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
export const CHAT_WIDTH_KEY = "local-lm-chat-width";
export const MOTION_KEY = "local-lm-motion";
export const THUMBNAIL_SIZE_KEY = "local-lm-thumbnails";

/** Whether the workspace animates: as the computer asks, or always as little as it can.
 *
 * "system" follows the operating system's reduced-motion setting, including
 * when it changes while the workspace is open. "reduce" reduces motion whatever
 * the computer says, for somebody who wants it here without changing it
 * everywhere else.
 */
export type MotionChoice = "system" | "reduce";

/** How much of the window the conversation and its composer may use.
 *
 * Standard is the column the workspace has always used, narrow enough that a
 * line of prose stays readable. Wide and full give tables, code and side-by-side
 * media room on a large screen, for somebody who would rather have that than
 * short lines.
 */
export type ChatWidth = "standard" | "wide" | "full";

export const CHAT_WIDTHS: readonly ChatWidth[] = ["standard", "wide", "full"];

/** How large media previews are drawn: Media Library cards and pictures attached to a message.
 *
 * Medium is the size they have always been. Small fits more of a large library
 * on screen; large makes a picture recognisable without opening it.
 */
export type ThumbnailSize = "small" | "medium" | "large";

export const THUMBNAIL_SIZES: readonly ThumbnailSize[] = ["small", "medium", "large"];

export function isRoom(value: unknown): value is Room {
  return typeof value === "string" && (ROOMS as readonly string[]).includes(value);
}

export function isMode(value: unknown): value is ThemeMode {
  return value === "light" || value === "dark";
}

export function isModeChoice(value: unknown): value is ModeChoice {
  return isMode(value) || value === "system";
}

export function isMotionChoice(value: unknown): value is MotionChoice {
  return value === "system" || value === "reduce";
}

const PREFERS_LESS_MOTION = "(prefers-reduced-motion: reduce)";

function systemPrefersLessMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia(PREFERS_LESS_MOTION).matches;
}

export function storedMotionChoice(): MotionChoice {
  const stored = localStorage.getItem(MOTION_KEY);
  return isMotionChoice(stored) ? stored : "system";
}

/** Whether motion should be reduced right now, for movement the stylesheet cannot reach.
 *
 * A script that scrolls with `behavior: "smooth"` animates whatever CSS says, so
 * it has to ask. This reads the remembered choice and the computer's setting
 * directly rather than the document attribute: a component's effect can run
 * before the one that sets the attribute, and the first scroll would then
 * animate for somebody who asked it not to.
 */
export function prefersLessMotion(): boolean {
  return storedMotionChoice() === "reduce" || systemPrefersLessMotion();
}

/** The motion choice, whether motion is reduced now, and a way to change the choice. */
export function useMotion(): [MotionChoice, boolean, (choice: MotionChoice) => void] {
  const [choice, setChoice] = useState<MotionChoice>(storedMotionChoice);
  const [system, setSystem] = useState<boolean>(systemPrefersLessMotion);
  useEffect(() => {
    if (choice !== "system" || typeof matchMedia !== "function") return;
    const query = matchMedia(PREFERS_LESS_MOTION);
    const follow = () => setSystem(query.matches);
    follow();
    query.addEventListener("change", follow);
    return () => query.removeEventListener("change", follow);
  }, [choice]);
  const choose = useCallback((next: MotionChoice) => {
    setChoice(next);
    localStorage.setItem(MOTION_KEY, next);
  }, []);
  return [choice, choice === "reduce" || system, choose];
}

export function isChatWidth(value: unknown): value is ChatWidth {
  return typeof value === "string" && (CHAT_WIDTHS as readonly string[]).includes(value);
}

export function storedChatWidth(): ChatWidth {
  const stored = localStorage.getItem(CHAT_WIDTH_KEY);
  return isChatWidth(stored) ? stored : "standard";
}

export function isThumbnailSize(value: unknown): value is ThumbnailSize {
  return typeof value === "string" && (THUMBNAIL_SIZES as readonly string[]).includes(value);
}

export function storedThumbnailSize(): ThumbnailSize {
  const stored = localStorage.getItem(THUMBNAIL_SIZE_KEY);
  return isThumbnailSize(stored) ? stored : "medium";
}

export function useChatWidth(): [ChatWidth, (width: ChatWidth) => void] {
  const [width, setWidth] = useState<ChatWidth>(storedChatWidth);
  const choose = useCallback((next: ChatWidth) => {
    setWidth(next);
    localStorage.setItem(CHAT_WIDTH_KEY, next);
  }, []);
  return [width, choose];
}

export function useThumbnailSize(): [ThumbnailSize, (size: ThumbnailSize) => void] {
  const [size, setSize] = useState<ThumbnailSize>(storedThumbnailSize);
  const choose = useCallback((next: ThumbnailSize) => {
    setSize(next);
    localStorage.setItem(THUMBNAIL_SIZE_KEY, next);
  }, []);
  return [size, choose];
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

/** The room, its light, the chat width and motion, remembered and applied to the document.
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
  chatWidth: ChatWidth;
  /** What was chosen for motion, which may be to follow the computer. */
  motionChoice: MotionChoice;
  thumbnailSize: ThumbnailSize;
  setRoom: (room: Room) => void;
  setMode: (choice: ModeChoice) => void;
  setChatWidth: (width: ChatWidth) => void;
  setMotion: (choice: MotionChoice) => void;
  setThumbnailSize: (size: ThumbnailSize) => void;
}

export function useAppearance(): Appearance {
  const [room, setRoom] = useRoom();
  const [modeChoice, mode, setMode] = useThemeMode();
  const [chatWidth, setChatWidth] = useChatWidth();
  const [motionChoice, reducedMotion, setMotion] = useMotion();
  const [thumbnailSize, setThumbnailSize] = useThumbnailSize();
  useEffect(() => {
    document.documentElement.dataset.room = room;
    document.documentElement.dataset.mode = mode;
    document.documentElement.dataset.chatWidth = chatWidth;
    document.documentElement.dataset.motion = reducedMotion ? "reduced" : "full";
    document.documentElement.dataset.thumbnails = thumbnailSize;
  }, [room, mode, chatWidth, reducedMotion, thumbnailSize]);
  return useMemo(
    () => ({
      room, mode, modeChoice, chatWidth, motionChoice, thumbnailSize,
      setRoom, setMode, setChatWidth, setMotion, setThumbnailSize,
    }),
    [room, mode, modeChoice, chatWidth, motionChoice, thumbnailSize, setRoom, setMode, setChatWidth, setMotion, setThumbnailSize],
  );
}
