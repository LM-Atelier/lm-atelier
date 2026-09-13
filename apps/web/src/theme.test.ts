import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  CHAT_WIDTH_KEY,
  MODE_KEY,
  ROOMS,
  ROOM_LABELS,
  isChatWidth,
  isMode,
  isModeChoice,
  isRoom,
  useAppearance,
  type Room,
} from "./theme";

/** An operating system whose light setting the test can switch, as a person would. */
function systemSetTo(light: boolean) {
  let current = light;
  const listeners = new Set<() => void>();
  const query = {
    media: "(prefers-color-scheme: light)",
    get matches() {
      return current;
    },
    addEventListener: (_type: string, listener: () => void) => listeners.add(listener),
    removeEventListener: (_type: string, listener: () => void) => listeners.delete(listener),
  };
  vi.stubGlobal("matchMedia", () => query);
  return {
    listeners,
    switchTo(next: boolean) {
      current = next;
      for (const listener of listeners) listener();
    },
  };
}

describe("rooms and modes", () => {
  it("treats the room and the light as separate questions", () => {
    // The old design derived the room from the view, so a dark sidebar sat
    // beside a paper chat and read as one interface disagreeing with itself.
    // Neither axis may be inferred from the other, or from what is on screen.
    expect(isRoom("north-light")).toBe(true);
    expect(isMode("light")).toBe(true);
    expect(isMode("dark")).toBe(true);
    expect(isMode("by-room")).toBe(false);
    expect(isRoom("light")).toBe(false);
  });

  it("names every room it offers", () => {
    // A room with no label cannot be chosen by anyone who is not reading
    // the source.
    for (const room of ROOMS) {
      expect(ROOM_LABELS[room as Room]).toBeTruthy();
    }
    expect(Object.keys(ROOM_LABELS).sort()).toEqual([...ROOMS].sort());
  });

  it("refuses anything that is not a room it ships", () => {
    expect(isRoom("solarized")).toBe(false);
    expect(isRoom("")).toBe(false);
    expect(isRoom(null)).toBe(false);
  });
});

describe("the appearance the workspace shares", () => {
  beforeEach(() => localStorage.clear());
  afterEach(cleanup);

  it("keeps its identity until the room or the light changes", () => {
    // Settings renders inside a memoized view. An appearance rebuilt on every
    // render would rebuild that whole view on every render as well.
    const { result, rerender } = renderHook(() => useAppearance());
    const first = result.current;

    rerender();
    expect(result.current).toBe(first);

    act(() => first.setMode(first.mode === "dark" ? "light" : "dark"));
    expect(result.current).not.toBe(first);
    expect(result.current.setMode).toBe(first.setMode);
    expect(result.current.setRoom).toBe(first.setRoom);
  });
});

describe("how wide the chat reads", () => {
  beforeEach(() => {
    localStorage.clear();
    delete document.documentElement.dataset.chatWidth;
  });
  afterEach(cleanup);

  it("starts at the standard column, and says so on the document", () => {
    const { result } = renderHook(() => useAppearance());

    expect(result.current.chatWidth).toBe("standard");
    expect(document.documentElement.dataset.chatWidth).toBe("standard");
  });

  it("applies and remembers a wider column, and a new identity only when it changes", () => {
    const { result } = renderHook(() => useAppearance());
    const first = result.current;

    act(() => first.setChatWidth("full"));

    expect(document.documentElement.dataset.chatWidth).toBe("full");
    expect(localStorage.getItem(CHAT_WIDTH_KEY)).toBe("full");
    expect(result.current).not.toBe(first);
    expect(result.current.setChatWidth).toBe(first.setChatWidth);
    // The width is its own question: the light and the room stayed where they were.
    expect(result.current.mode).toBe(first.mode);
    expect(result.current.room).toBe(first.room);
  });

  it("opens on the remembered width, and on standard for anything it does not offer", () => {
    localStorage.setItem(CHAT_WIDTH_KEY, "wide");
    expect(renderHook(() => useAppearance()).result.current.chatWidth).toBe("wide");
    cleanup();

    localStorage.setItem(CHAT_WIDTH_KEY, "enormous");
    expect(renderHook(() => useAppearance()).result.current.chatWidth).toBe("standard");
    expect(["standard", "wide", "full"].every(isChatWidth)).toBe(true);
    expect(isChatWidth("enormous")).toBe(false);
  });
});

describe("following the system's light", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("knows the system as a choice, and nothing else", () => {
    expect(isModeChoice("system")).toBe(true);
    expect(isModeChoice("dark")).toBe(true);
    expect(isModeChoice("auto")).toBe(false);
    expect(isMode("system")).toBe(false);
  });

  it("follows the computer while System is chosen, and is remembered as that choice", () => {
    const system = systemSetTo(false);
    const { result } = renderHook(() => useAppearance());

    act(() => result.current.setMode("system"));
    expect(result.current.modeChoice).toBe("system");
    expect(result.current.mode).toBe("dark");
    expect(localStorage.getItem(MODE_KEY)).toBe("system");

    act(() => system.switchTo(true));
    expect(result.current.mode).toBe("light");
    expect(document.documentElement.dataset.mode).toBe("light");
  });

  it("stops listening once a fixed mode is chosen again", () => {
    const system = systemSetTo(true);
    const { result } = renderHook(() => useAppearance());
    act(() => result.current.setMode("system"));
    expect(system.listeners.size).toBe(1);

    act(() => result.current.setMode("dark"));
    expect(system.listeners.size).toBe(0);
    act(() => system.switchTo(false));
    act(() => system.switchTo(true));
    expect(result.current.mode).toBe("dark");
  });

  it("without any choice, seeds once from the system and then stays put", () => {
    // Following the system is something a person chooses. Nobody chose it here,
    // so a later change in the computer's setting does not repaint the workspace.
    const system = systemSetTo(true);
    const { result } = renderHook(() => useAppearance());
    expect(result.current.mode).toBe("light");
    expect(result.current.modeChoice).toBe("light");

    act(() => system.switchTo(false));
    expect(result.current.mode).toBe("light");
    expect(system.listeners.size).toBe(0);
  });

  it("opens on System when that was the remembered choice", () => {
    localStorage.setItem(MODE_KEY, "system");
    systemSetTo(true);

    const { result } = renderHook(() => useAppearance());

    expect(result.current.modeChoice).toBe("system");
    expect(result.current.mode).toBe("light");
  });
});
