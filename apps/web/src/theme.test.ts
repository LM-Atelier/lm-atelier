import { describe, expect, it } from "vitest";

import { ROOMS, ROOM_LABELS, isMode, isRoom, type Room } from "./theme";

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
