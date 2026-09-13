import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  CHAT_WIDTH_KEY,
  MODE_KEY,
  MOTION_KEY,
  ROOMS,
  ROOM_LABELS,
  TEXT_SIZE_KEY,
  THUMBNAIL_SIZE_KEY,
  isChatWidth,
  isMode,
  isModeChoice,
  isMotionChoice,
  isRoom,
  isTextSize,
  isThumbnailSize,
  prefersLessMotion,
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
  // Only the light query is this one. A real browser answers each media query
  // with its own object, so another query's listeners must not count as these.
  vi.stubGlobal("matchMedia", (media: string) =>
    media === query.media
      ? query
      : { media, matches: false, addEventListener: () => undefined, removeEventListener: () => undefined },
  );
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

describe("how large text is", () => {
  beforeEach(() => {
    localStorage.clear();
    delete document.documentElement.dataset.textSize;
  });
  afterEach(cleanup);

  it("starts at standard, the size text has always been, and says so on the document", () => {
    const { result } = renderHook(() => useAppearance());

    expect(result.current.textSize).toBe("standard");
    expect(document.documentElement.dataset.textSize).toBe("standard");
  });

  it("applies and remembers a larger size without moving anything else", () => {
    const { result } = renderHook(() => useAppearance());
    const first = result.current;

    act(() => first.setTextSize("larger"));

    expect(document.documentElement.dataset.textSize).toBe("larger");
    expect(localStorage.getItem(TEXT_SIZE_KEY)).toBe("larger");
    expect(result.current.setTextSize).toBe(first.setTextSize);
    expect(result.current.thumbnailSize).toBe(first.thumbnailSize);
    expect(result.current.chatWidth).toBe(first.chatWidth);
  });

  it("opens on the remembered size, and on standard for anything it does not offer", () => {
    localStorage.setItem(TEXT_SIZE_KEY, "large");
    expect(renderHook(() => useAppearance()).result.current.textSize).toBe("large");
    cleanup();

    localStorage.setItem(TEXT_SIZE_KEY, "gigantic");
    expect(renderHook(() => useAppearance()).result.current.textSize).toBe("standard");
    expect(["standard", "large", "larger"].every(isTextSize)).toBe(true);
    expect(isTextSize("gigantic")).toBe(false);
  });
});

describe("how large thumbnails are", () => {
  beforeEach(() => {
    localStorage.clear();
    delete document.documentElement.dataset.thumbnails;
  });
  afterEach(cleanup);

  it("starts at medium, the size they have always been, and says so on the document", () => {
    const { result } = renderHook(() => useAppearance());

    expect(result.current.thumbnailSize).toBe("medium");
    expect(document.documentElement.dataset.thumbnails).toBe("medium");
  });

  it("applies and remembers a size without moving anything else", () => {
    const { result } = renderHook(() => useAppearance());
    const first = result.current;

    act(() => first.setThumbnailSize("large"));

    expect(document.documentElement.dataset.thumbnails).toBe("large");
    expect(localStorage.getItem(THUMBNAIL_SIZE_KEY)).toBe("large");
    expect(result.current.setThumbnailSize).toBe(first.setThumbnailSize);
    expect(result.current.chatWidth).toBe(first.chatWidth);
    expect(result.current.mode).toBe(first.mode);
  });

  it("opens on the remembered size, and on medium for anything it does not offer", () => {
    localStorage.setItem(THUMBNAIL_SIZE_KEY, "small");
    expect(renderHook(() => useAppearance()).result.current.thumbnailSize).toBe("small");
    cleanup();

    localStorage.setItem(THUMBNAIL_SIZE_KEY, "huge");
    expect(renderHook(() => useAppearance()).result.current.thumbnailSize).toBe("medium");
    expect(["small", "medium", "large"].every(isThumbnailSize)).toBe(true);
    expect(isThumbnailSize("huge")).toBe(false);
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

/** A computer whose reduced-motion setting the test can switch; every other query answers no. */
function motionSetTo(reduced: boolean) {
  let current = reduced;
  const listeners = new Set<() => void>();
  vi.stubGlobal("matchMedia", (media: string) => ({
    media,
    get matches() {
      return media === "(prefers-reduced-motion: reduce)" && current;
    },
    addEventListener: (_type: string, listener: () => void) => listeners.add(listener),
    removeEventListener: (_type: string, listener: () => void) => listeners.delete(listener),
  }));
  return {
    listeners,
    switchTo(next: boolean) {
      current = next;
      for (const listener of listeners) listener();
    },
  };
}

describe("reducing motion", () => {
  beforeEach(() => {
    localStorage.clear();
    delete document.documentElement.dataset.motion;
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("follows the computer's setting while Automatic, including when it changes", () => {
    const computer = motionSetTo(false);
    const { result } = renderHook(() => useAppearance());
    expect(result.current.motionChoice).toBe("system");
    expect(document.documentElement.dataset.motion).toBe("full");

    act(() => computer.switchTo(true));

    expect(document.documentElement.dataset.motion).toBe("reduced");
    expect(prefersLessMotion()).toBe(true);
  });

  it("reduces whatever the computer says once Reduce is chosen, and remembers it", () => {
    const computer = motionSetTo(false);
    const { result } = renderHook(() => useAppearance());

    act(() => result.current.setMotion("reduce"));

    expect(document.documentElement.dataset.motion).toBe("reduced");
    expect(localStorage.getItem(MOTION_KEY)).toBe("reduce");
    expect(computer.listeners.size).toBe(0);
    expect(prefersLessMotion()).toBe(true);
    // The light and the width are separate questions and did not move.
    expect(document.documentElement.dataset.chatWidth).toBe("standard");
  });

  it("answers from the choice and the computer before any hook has run", () => {
    // A scroll can happen in an effect that runs before the one that marks the
    // document, so the answer cannot depend on that mark.
    motionSetTo(false);
    expect(prefersLessMotion()).toBe(false);
    localStorage.setItem(MOTION_KEY, "reduce");
    expect(prefersLessMotion()).toBe(true);
    localStorage.setItem(MOTION_KEY, "system");
    motionSetTo(true);
    expect(prefersLessMotion()).toBe(true);
    expect(document.documentElement.dataset.motion).toBeUndefined();
  });

  it("knows only Automatic and Reduce as choices", () => {
    expect(isMotionChoice("system")).toBe(true);
    expect(isMotionChoice("reduce")).toBe(true);
    expect(isMotionChoice("none")).toBe(false);
    localStorage.setItem(MOTION_KEY, "none");
    motionSetTo(false);
    expect(renderHook(() => useAppearance()).result.current.motionChoice).toBe("system");
  });
});
