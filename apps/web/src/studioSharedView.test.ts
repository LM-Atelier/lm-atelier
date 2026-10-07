/** One view shared by the two halves of a side-by-side comparison. */

import { describe, expect, it } from "vitest";
import { halfViewport, panSharedBy, WHOLE_VIEW, zoomSharedAbout } from "./studioSharedView";
import { shownRect, toImagePoint } from "./studioViewport";

const HALF = { width: 200, height: 200 };
const SMALL = { width: 400, height: 200 };
// The same shape at twice the pixels: a result made larger shows the same scene.
const LARGE = { width: 800, height: 400 };

describe("the view both halves share", () => {
  it("starts with each picture whole and centred in its half", () => {
    expect(shownRect(halfViewport(WHOLE_VIEW, SMALL, HALF), SMALL)).toEqual({ x: 0, y: 50, width: 200, height: 100 });
  });

  it("shows two pictures of one shape at the same place and size, whatever their pixels", () => {
    expect(shownRect(halfViewport(WHOLE_VIEW, LARGE, HALF), LARGE))
      .toEqual(shownRect(halfViewport(WHOLE_VIEW, SMALL, HALF), SMALL));
  });

  it("zooms about the point under the pointer on one half, and the other half follows it there", () => {
    const anchor = { x: 50, y: 110 };
    const before = toImagePoint(halfViewport(WHOLE_VIEW, SMALL, HALF), anchor);

    const zoomed = zoomSharedAbout(WHOLE_VIEW, SMALL, HALF, anchor, 2);

    const small = toImagePoint(halfViewport(zoomed, SMALL, HALF), anchor);
    const large = toImagePoint(halfViewport(zoomed, LARGE, HALF), anchor);
    expect(zoomed.zoom).toBeCloseTo(2);
    expect(small.x).toBeCloseTo(before.x);
    expect(small.y).toBeCloseTo(before.y);
    // The same share of the other picture sits under the pointer in its half.
    expect(large.x / LARGE.width).toBeCloseTo(before.x / SMALL.width);
    expect(large.y / LARGE.height).toBeCloseTo(before.y / SMALL.height);
  });

  it("moves both halves together when either is dragged", () => {
    const zoomed = zoomSharedAbout(WHOLE_VIEW, SMALL, HALF, { x: 100, y: 100 }, 3);

    const moved = panSharedBy(zoomed, LARGE, HALF, 30, -12);

    for (const picture of [SMALL, LARGE]) {
      const was = shownRect(halfViewport(zoomed, picture, HALF), picture);
      const now = shownRect(halfViewport(moved, picture, HALF), picture);
      expect(now.x - was.x).toBeCloseTo(30);
      expect(now.y - was.y).toBeCloseTo(-12);
      expect(now.width).toBeCloseTo(was.width);
    }
  });

  it("keeps every number finite before the halves have been measured", () => {
    const view = panSharedBy(zoomSharedAbout(WHOLE_VIEW, SMALL, { width: 0, height: 0 }, { x: 0, y: 0 }, 2), SMALL, { width: 0, height: 0 }, 5, 5);

    expect([view.zoom, view.x, view.y].every(Number.isFinite)).toBe(true);
  });
});
