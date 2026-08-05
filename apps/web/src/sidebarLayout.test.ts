import { beforeEach, describe, expect, it } from "vitest";

import {
  DEFAULT_SIDEBAR_WIDTH,
  MAX_SIDEBAR_WIDTH,
  MIN_SIDEBAR_WIDTH,
  clampSidebarWidth,
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
