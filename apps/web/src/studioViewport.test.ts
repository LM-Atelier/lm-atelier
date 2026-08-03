import { describe, expect, it } from "vitest";
import {
  fitViewport,
  identityViewport,
  MAX_SCALE,
  MIN_SCALE,
  panBy,
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
});
