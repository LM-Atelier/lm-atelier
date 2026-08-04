/** Who decides the light, and what wins. */

import { describe, expect, it } from "vitest";
import { isThemeChoice, roomFor } from "./theme";

describe("appearance choice", () => {
  it("lets each surface pick its own room by default", () => {
    // The design's own answer: paper where you read, dark where you judge
    // pictures, because neither is right for the other's work.
    expect(roomFor("by-room", true)).toBe("reading");
    expect(roomFor("by-room", false)).toBe("making");
  });

  it("lets a person overrule the design", () => {
    // Someone working at night wants the whole thing dark whatever prose
    // prefers, and picking light means the studio is on paper too - worse
    // for judging colour, and still what was asked for.
    expect(roomFor("dark", true)).toBe("making");
    expect(roomFor("light", false)).toBe("reading");
  });

  it("falls back to the default rather than trusting stored junk", () => {
    expect(isThemeChoice("light")).toBe(true);
    expect(isThemeChoice("sepia")).toBe(false);
    expect(isThemeChoice(null)).toBe(false);
  });
});
