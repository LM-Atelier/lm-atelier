import { describe, expect, it } from "vitest";
import {
  EXTEND_SHAPES,
  extendedFrame,
  extendedSize,
  marginPixels,
  marginsForShape,
  marginsForSize,
  type ExtendShape,
} from "./studioExtend";

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

function shape(label: string): ExtendShape {
  const found = EXTEND_SHAPES.find((candidate) => candidate.label === label);
  if (!found) throw new Error(`no shape ${label}`);
  return found;
}

describe("extending to a shape", () => {
  it("grows a wider picture taller and a taller one wider, evenly on the edges that grow", () => {
    // 400 by 300 is wider than a square, so it gains 50 above and 50 below.
    const wide = { width: 400, height: 300 };
    expect(marginPixels(marginsForShape(wide, shape("Square"))!, wide)).toEqual({ top: 50, right: 0, bottom: 50, left: 0 });
    // 300 by 400 is narrower than 16:9, which needs 711 across (400 * 16 / 9 is
    // 711.1): 205 more on the left and 206 on the right.
    const tall = { width: 300, height: 400 };
    expect(marginPixels(marginsForShape(tall, shape("16:9"))!, tall)).toEqual({ top: 0, right: 206, bottom: 0, left: 205 });
    // 100 * 16 / 9 is 177.8, so a 100 by 100 picture goes to 178 across, not 177.
    const square = { width: 100, height: 100 };
    expect(extendedSize(marginsForShape(square, shape("16:9"))!, square)).toEqual({ width: 178, height: 100 });
  });

  it("changes nothing on a picture that is the shape already", () => {
    expect(marginsForShape({ width: 1600, height: 900 }, shape("16:9"))).toEqual(NONE);
    expect(marginsForShape({ width: 512, height: 512 }, shape("Square"))).toEqual(NONE);
  });

  it("offers nothing that needs more than twice the picture on an edge", () => {
    // 1000 by 100 needs 1778 down for 9:16: 839 above, over eight times its height.
    expect(marginsForShape({ width: 1000, height: 100 }, shape("9:16"))).toBeNull();
  });

  it("comes within half a pixel of the shape through the server's rounding, never cutting", () => {
    for (const target of EXTEND_SHAPES) {
      for (const width of [1, 7, 99, 100, 333, 1024, 4000]) {
        for (const height of [1, 9, 100, 160, 575, 768, 3000]) {
          const picture = { width, height };
          const margins = marginsForShape(picture, target);
          if (!margins) continue;
          const gained = marginPixels(margins, picture);
          const size = extendedSize(margins, picture);
          // One pair of edges grows, the two within a pixel of each other.
          expect((gained.top === 0 && gained.bottom === 0) || (gained.left === 0 && gained.right === 0)).toBe(true);
          expect(Math.abs(gained.left - gained.right)).toBeLessThanOrEqual(1);
          expect(Math.abs(gained.top - gained.bottom)).toBeLessThanOrEqual(1);
          // The grown side is the whole pixel nearest the shape: within half a
          // pixel of it, which in these whole numbers is half the other side's term.
          const off = Math.abs(size.width * target.height - size.height * target.width);
          if (gained.left + gained.right > 0) expect(off).toBeLessThanOrEqual(target.height / 2);
          if (gained.top + gained.bottom > 0) expect(off).toBeLessThanOrEqual(target.width / 2);
          expect(size.width).toBeGreaterThanOrEqual(width);
          expect(size.height).toBeGreaterThanOrEqual(height);
        }
      }
    }
  });
});

describe("extending to a size", () => {
  const picture = { width: 100, height: 160 };
  const size = { width: 150, height: 200 };

  it("puts the new room away from where the picture is anchored", () => {
    expect(marginPixels(marginsForSize(picture, size, "center")!, picture)).toEqual({ top: 20, right: 25, bottom: 20, left: 25 });
    expect(marginPixels(marginsForSize(picture, size, "top_left")!, picture)).toEqual({ top: 0, right: 50, bottom: 40, left: 0 });
    expect(marginPixels(marginsForSize(picture, size, "bottom_right")!, picture)).toEqual({ top: 40, right: 0, bottom: 0, left: 50 });
    expect(marginPixels(marginsForSize(picture, size, "left")!, picture)).toEqual({ top: 20, right: 50, bottom: 20, left: 0 });
    expect(extendedSize(marginsForSize(picture, size, "top")!, picture)).toEqual(size);
  });

  it("gives an odd pixel to the right and the bottom", () => {
    expect(marginPixels(marginsForSize(picture, { width: 101, height: 161 }, "center")!, picture)).toEqual({
      top: 0,
      right: 1,
      bottom: 1,
      left: 0,
    });
  });

  it("refuses a smaller size, and one needing more than twice the picture on an edge", () => {
    expect(marginsForSize(picture, { width: 99, height: 200 }, "center")).toBeNull();
    expect(marginsForSize(picture, { width: 150, height: 159 }, "center")).toBeNull();
    // Anchored at the left, all 201 new pixels go to the right: past twice 100.
    expect(marginsForSize(picture, { width: 301, height: 160 }, "left")).toBeNull();
    // Centered, the same size is 100 on the left and 101 on the right.
    expect(marginPixels(marginsForSize(picture, { width: 301, height: 160 }, "center")!, picture)).toEqual({
      top: 0,
      right: 101,
      bottom: 0,
      left: 100,
    });
  });
});
