import { describe, expect, it } from "vitest";
import {
  fitViewport,
  identityViewport,
  MAX_SCALE,
  MIN_SCALE,
  panBy,
  pinchBy,
  shownRect,
  toImagePoint,
  toScreenPoint,
  zoomAbout,
} from "./studioViewport";

describe("studio viewport", () => {
  it("round-trips points through the transform", () => {
    const viewport = { scale: 2.5, tx: 40, ty: -12 };
    const image = { x: 123.5, y: 67.25 };
    const back = toImagePoint(viewport, toScreenPoint(viewport, image));
    expect(back.x).toBeCloseTo(image.x);
    expect(back.y).toBeCloseTo(image.y);
  });

  it("keeps the pixel under the cursor fixed while zooming", () => {
    let viewport = identityViewport();
    const anchor = { x: 300, y: 200 };
    const before = toImagePoint(viewport, anchor);
    viewport = zoomAbout(viewport, anchor, 2);
    viewport = zoomAbout(viewport, anchor, 1.5);
    const after = toImagePoint(viewport, anchor);
    expect(after.x).toBeCloseTo(before.x);
    expect(after.y).toBeCloseTo(before.y);
  });

  it("clamps zoom to the supported range", () => {
    let viewport = identityViewport();
    viewport = zoomAbout(viewport, { x: 0, y: 0 }, 1e-9);
    expect(viewport.scale).toBe(MIN_SCALE);
    viewport = zoomAbout(viewport, { x: 0, y: 0 }, 1e9);
    expect(viewport.scale).toBe(MAX_SCALE);
  });

  it("pans additively", () => {
    const viewport = panBy(panBy(identityViewport(), 10, -5), -4, 3);
    expect(viewport.tx).toBe(6);
    expect(viewport.ty).toBe(-2);
  });

  it("follows two pointers: scaled by their spread, moved with the point between them", () => {
    const start = { scale: 1.5, tx: 20, ty: -10 };
    const fingers = [{ x: 100, y: 120 }, { x: 180, y: 120 }] as const;
    const spread = [{ x: 60, y: 140 }, { x: 220, y: 140 }] as const;
    const under = fingers.map((finger) => toImagePoint(start, finger));

    const viewport = pinchBy(start, fingers, spread);

    // Twice as far apart, so twice the zoom; the line between them kept its
    // direction, so what lay under each finger is under it still.
    expect(viewport.scale).toBeCloseTo(3);
    spread.forEach((finger, index) => {
      const now = toImagePoint(viewport, finger);
      expect(now.x).toBeCloseTo(under[index].x);
      expect(now.y).toBeCloseTo(under[index].y);
    });
  });

  it("pans without zooming when two pointers move together", () => {
    const viewport = pinchBy(
      { scale: 2, tx: 0, ty: 0 },
      [{ x: 10, y: 10 }, { x: 50, y: 30 }],
      [{ x: 25, y: 5 }, { x: 65, y: 25 }],
    );
    expect(viewport).toEqual({ scale: 2, tx: 15, ty: -5 });
  });

  it("keeps what lay midway between two twisting pointers midway between them", () => {
    const viewport = pinchBy(
      identityViewport(),
      [{ x: 100, y: 100 }, { x: 200, y: 100 }],
      [{ x: 170, y: 40 }, { x: 170, y: 200 }],
    );
    expect(viewport.scale).toBeCloseTo(1.6);
    const middle = toImagePoint(viewport, { x: 170, y: 120 });
    expect(middle.x).toBeCloseTo(150);
    expect(middle.y).toBeCloseTo(100);
  });

  it("stops at the zoom limit with what lay midway still midway", () => {
    const start = { scale: MAX_SCALE / 2, tx: 5, ty: 5 };
    const middle = toImagePoint(start, { x: 100, y: 50 });
    // Ten times as far apart, about the same middle.
    const viewport = pinchBy(start, [{ x: 90, y: 50 }, { x: 110, y: 50 }], [{ x: 0, y: 50 }, { x: 200, y: 50 }]);
    expect(viewport.scale).toBe(MAX_SCALE);
    const now = toImagePoint(viewport, { x: 100, y: 50 });
    expect(now.x).toBeCloseTo(middle.x);
    expect(now.y).toBeCloseTo(middle.y);
  });

  it("moves but does not scale when the pointers start or end on one spot", () => {
    expect(pinchBy(identityViewport(), [{ x: 50, y: 50 }, { x: 50, y: 50 }], [{ x: 40, y: 60 }, { x: 80, y: 60 }]))
      .toEqual({ scale: 1, tx: 10, ty: 10 });
    expect(pinchBy(identityViewport(), [{ x: 40, y: 60 }, { x: 80, y: 60 }], [{ x: 60, y: 60 }, { x: 60, y: 60 }]))
      .toEqual({ scale: 1, tx: 0, ty: 0 });
  });

  it("fits and centers the image in the container", () => {
    const viewport = fitViewport({ width: 2000, height: 1000 }, { width: 800, height: 800 });
    expect(viewport.scale).toBeCloseTo(0.4);
    expect(viewport.tx).toBeCloseTo(0);
    // 1000 * 0.4 = 400 tall, centered in 800.
    expect(viewport.ty).toBeCloseTo(200);
    const image = toImagePoint(viewport, { x: 400, y: 400 });
    expect(image.x).toBeCloseTo(1000);
    expect(image.y).toBeCloseTo(500);
  });

  it("says where the picture is shown and how big, at the current zoom", () => {
    const fitted = fitViewport({ width: 2000, height: 1000 }, { width: 800, height: 800 });
    const shown = shownRect(fitted, { width: 2000, height: 1000 });
    expect(shown.x).toBeCloseTo(0);
    expect(shown.y).toBeCloseTo(200);
    expect(shown.width).toBeCloseTo(800);
    expect(shown.height).toBeCloseTo(400);

    // Zooming about the middle grows the picture on screen around that point.
    const zoomed = shownRect(zoomAbout(fitted, { x: 400, y: 400 }, 2), { width: 2000, height: 1000 });
    expect(zoomed.x).toBeCloseTo(-400);
    expect(zoomed.y).toBeCloseTo(0);
    expect(zoomed.width).toBeCloseTo(1600);
    expect(zoomed.height).toBeCloseTo(800);
  });
});
