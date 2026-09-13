import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  DEFAULT_SIDEBAR_WIDTH,
  MAX_SIDEBAR_WIDTH,
  MIN_SIDEBAR_WIDTH,
  SIDEBAR_COLLAPSED_KEY,
  SIDEBAR_WIDTH_KEY,
  clampSidebarWidth,
  resetSidebarLayout,
  useSidebarLayout,
  useSidebarState,
} from "./sidebarLayout";

describe("sidebar width", () => {
  beforeEach(() => localStorage.clear());

  it("cannot be dragged narrower than it can be read or wider than the work", () => {
    // A drag reports a raw pointer position, which is happily negative when
    // the pointer leaves the window to the left.
    expect(clampSidebarWidth(-400)).toBe(MIN_SIDEBAR_WIDTH);
    expect(clampSidebarWidth(0)).toBe(MIN_SIDEBAR_WIDTH);
    expect(clampSidebarWidth(9000)).toBe(MAX_SIDEBAR_WIDTH);
  });

  it("keeps a width inside the bounds exactly as given", () => {
    expect(clampSidebarWidth(300)).toBe(300);
    expect(clampSidebarWidth(MIN_SIDEBAR_WIDTH)).toBe(MIN_SIDEBAR_WIDTH);
    expect(clampSidebarWidth(MAX_SIDEBAR_WIDTH)).toBe(MAX_SIDEBAR_WIDTH);
  });

  it("falls back rather than storing a width that is not a number", () => {
    // localStorage returns strings, and a corrupted one must not become NaN
    // pixels - which resolves to no column at all.
    expect(clampSidebarWidth(Number.NaN)).toBe(DEFAULT_SIDEBAR_WIDTH);
    expect(clampSidebarWidth(Number.POSITIVE_INFINITY)).toBe(DEFAULT_SIDEBAR_WIDTH);
  });

  it("rounds to whole pixels", () => {
    expect(clampSidebarWidth(300.6)).toBe(301);
  });
});

describe("sidebar layout", () => {
  beforeEach(() => localStorage.clear());
  afterEach(cleanup);

  it("keeps every reader in step when the width or visibility changes", () => {
    const workspace = renderHook(() => useSidebarLayout());
    const settings = renderHook(() => useSidebarState());

    act(() => workspace.result.current.setWidth(330));
    act(() => workspace.result.current.toggle());

    expect(settings.result.current).toEqual({ width: 330, collapsed: true });
    expect(localStorage.getItem(SIDEBAR_WIDTH_KEY)).toBe("330");
    expect(localStorage.getItem(SIDEBAR_COLLAPSED_KEY)).toBe("true");
  });

  it("resets to shown at the usual width, and the workspace follows", () => {
    localStorage.setItem(SIDEBAR_WIDTH_KEY, "480");
    localStorage.setItem(SIDEBAR_COLLAPSED_KEY, "true");
    const workspace = renderHook(() => useSidebarLayout());
    expect(document.documentElement.style.getPropertyValue("--sidebar-width")).toBe("0px");

    act(() => resetSidebarLayout());

    expect(workspace.result.current.width).toBe(DEFAULT_SIDEBAR_WIDTH);
    expect(workspace.result.current.collapsed).toBe(false);
    expect(document.documentElement.style.getPropertyValue("--sidebar-width")).toBe(`${DEFAULT_SIDEBAR_WIDTH}px`);
    expect(localStorage.getItem(SIDEBAR_WIDTH_KEY)).toBeNull();
  });

  it("follows a reset made in another window", () => {
    localStorage.setItem(SIDEBAR_COLLAPSED_KEY, "true");
    const settings = renderHook(() => useSidebarState());
    expect(settings.result.current.collapsed).toBe(true);

    act(() => {
      localStorage.removeItem(SIDEBAR_COLLAPSED_KEY);
      window.dispatchEvent(new StorageEvent("storage", { key: SIDEBAR_COLLAPSED_KEY }));
    });

    expect(settings.result.current.collapsed).toBe(false);
  });
});
