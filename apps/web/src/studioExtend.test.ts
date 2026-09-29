import { describe, expect, it } from "vitest";
import { extendedFrame, extendedSize, marginPixels } from "./studioExtend";

const NONE = { top: 0, right: 0, bottom: 0, left: 0 };
const PICTURE = { width: 400, height: 200 };

describe("the size an extension makes", () => {
  it("takes the left and right edges from the width and the top and bottom from the height", () => {
    expect(marginPixels({ top: 0.5, right: 0.25, bottom: 0, left: 0.1 }, PICTURE)).toEqual({
      top: 100,
      right: 100,
      bottom: 0,
      left: 40,
    });
  });

  it("rounds half a pixel up and less than half down, as the server does", () => {
    // A quarter of 6 is 1.5 and a quarter of 5 is 1.25.
    expect(marginPixels({ ...NONE, right: 0.25, top: 0.25 }, { width: 6, height: 5 })).toEqual({
      top: 1,
      right: 2,
      bottom: 0,
      left: 0,
    });
  });

  it("adds every edge's pixels to the picture", () => {
    expect(extendedSize({ top: 0.5, right: 0.25, bottom: 0, left: 0.1 }, PICTURE)).toEqual({
      width: 540,
      height: 300,
    });
  });
});

describe("the frame an extension draws", () => {
  it("grows around the picture as shown, by what each edge gains at this zoom", () => {
    // Shown at twice its size, 50 pixels across and 20 down the canvas.
    const shown = { x: 50, y: 20, width: 800, height: 400 };

    expect(extendedFrame({ ...NONE, right: 0.25, top: 0.5 }, PICTURE, shown)).toEqual({
      x: 50,
      y: -180,
      width: 1000,
      height: 600,
    });
  });

  it("is the picture as shown when no edge moves", () => {
    const shown = { x: 12, y: 8, width: 200, height: 100 };

    expect(extendedFrame(NONE, PICTURE, shown)).toEqual(shown);
  });
});
