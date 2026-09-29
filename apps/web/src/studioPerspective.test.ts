/** Correcting the perspective: the corners judged as the server judges them, and dragging them. */

import { describe, expect, it, vi } from "vitest";
import {
  CORNERS,
  correctedSize,
  isFrame,
  isUnchanged,
  PerspectiveTool,
  pictureCorners,
  projected,
  thirds,
} from "./studioPerspective";
import type { ImagePoint } from "./studioTools";
import { initialToolState, snapshotBeforeGesture, studioToolReducer, toolFor } from "./studioToolState";
import type { StudioPerspective } from "./types";

/** Four corners from the top left, going clockwise. */
function corners(points: Array<[number, number]>): StudioPerspective {
  return {
    top_left: { x: points[0][0], y: points[0][1] },
    top_right: { x: points[1][0], y: points[1][1] },
    bottom_right: { x: points[2][0], y: points[2][1] },
    bottom_left: { x: points[3][0], y: points[3][1] },
  };
}

// The card test_studio_local_edits.py corrects, seen at an angle on a 120 by 90 picture.
const CARD = corners([[30, 12], [96, 22], [100, 80], [18, 70]]);
const SIZE = { width: 120, height: 90 };

describe("the corners", () => {
  it("make the size the server makes: the longer of each pair of opposite sides", () => {
    // The bottom is the square root of 82 squared plus 10 squared, 82.6, and
    // the left of 12 and 58, 59.2; the server's test asks for 83 by 59.
    expect(correctedSize(CARD)).toEqual({ width: 83, height: 59 });
    expect(correctedSize(pictureCorners(120, 90))).toEqual({ width: 120, height: 90 });
  });

  it("start at the picture's own, which corrects nothing", () => {
    expect(isUnchanged(pictureCorners(120, 90), 120, 90)).toBe(true);
    expect(isUnchanged(CARD, 120, 90)).toBe(false);
  });

  it("go around a card seen at an angle, as they go around the picture itself", () => {
    expect(isFrame(CARD)).toBe(true);
    expect(isFrame(pictureCorners(120, 90))).toBe(true);
  });

  it.each([
    ["the top corners swapped", [[96, 22], [30, 12], [100, 80], [18, 70]]],
    ["gone round the other way", [[30, 12], [18, 70], [100, 80], [96, 22]]],
    ["one pulled inside the shape", [[30, 12], [96, 22], [50, 30], [18, 70]]],
  ] as Array<[string, Array<[number, number]>]>)("are refused as the server refuses them: %s", (_name, points) => {
    expect(isFrame(corners(points))).toBe(false);
  });
});

describe("where the corrected picture falls on the source", () => {
  it("puts its corners on the four corners", () => {
    const ends = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([across, down]) => projected(CARD, across, down));
    CORNERS.forEach((name, index) => {
      expect(ends[index].x).toBeCloseTo(CARD[name].x, 9);
      expect(ends[index].y).toBeCloseTo(CARD[name].y, 9);
    });
  });

  it("puts its middle where the diagonals cross, as a view in perspective does", () => {
    const { top_left: a, top_right: b, bottom_right: c, bottom_left: d } = CARD;
    const cross = (u: ImagePoint, v: ImagePoint) => u.x * v.y - u.y * v.x;
    const minus = (u: ImagePoint, v: ImagePoint) => ({ x: u.x - v.x, y: u.y - v.y });
    const along = cross(minus(b, a), minus(d, b)) / cross(minus(c, a), minus(d, b));
    const crossing = { x: a.x + along * (c.x - a.x), y: a.y + along * (c.y - a.y) };

    const middle = projected(CARD, 0.5, 0.5);

    expect(middle.x).toBeCloseTo(crossing.x, 9);
    expect(middle.y).toBeCloseTo(crossing.y, 9);
    // Not the corners' average, which a straight blend of the sides would give:
    // that lies three pixels off here.
    expect(Math.hypot(middle.x - 61, middle.y - 46)).toBeGreaterThan(3);
  });

  it("divides the picture's own frame in even thirds", () => {
    const lines = thirds(pictureCorners(120, 90)).map((line) => line.map(({ x, y }) => [Math.round(x), Math.round(y)]));

    expect(lines).toEqual([
      [[40, 0], [40, 90]],
      [[0, 30], [120, 30]],
      [[80, 0], [80, 90]],
      [[0, 60], [120, 60]],
    ]);
  });
});

describe("dragging the corners", () => {
  it("moves the nearest corner as far as the pointer moves, never to where the press landed", () => {
    const onChange = vi.fn();
    const tool = new PerspectiveTool(pictureCorners(120, 90), SIZE, onChange);

    // Ten across and six down from the top right corner.
    tool.down({ x: 110, y: 6 });
    tool.move({ x: 100, y: 16 });
    expect(tool.preview()).toMatchObject({ kind: "corners", active: "top_right", corners: { top_right: { x: 110, y: 10 } } });
    tool.up({ x: 95.4, y: 20.6 });

    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith({ ...pictureCorners(120, 90), top_right: { x: 105, y: 15 } });
  });

  it("keeps each corner on the picture", () => {
    const onChange = vi.fn();
    const tool = new PerspectiveTool(CARD, SIZE, onChange);

    tool.down({ x: 20, y: 72 });
    tool.up({ x: -60, y: 200 });

    expect(onChange).toHaveBeenCalledWith({ ...CARD, bottom_left: { x: 0, y: 90 } });
  });

  it("puts a corner back when the drag is abandoned, and reports nothing", () => {
    const onChange = vi.fn();
    const tool = new PerspectiveTool(CARD, SIZE, onChange);

    tool.down({ x: 31, y: 12 });
    tool.move({ x: 60, y: 40 });
    tool.cancel();
    tool.up({ x: 60, y: 40 });

    expect(tool.preview()).toMatchObject({ corners: CARD });
    expect(onChange).not.toHaveBeenCalled();
  });

  it("marks the corner the pointer is nearest before it presses, and never touches the selection", () => {
    const tool = new PerspectiveTool(CARD, SIZE, vi.fn());

    tool.move({ x: 97, y: 76 });

    expect(tool.preview()).toMatchObject({ kind: "corners", active: "bottom_right" });
    expect(tool.appliesWhileMoving).toBe(false);
    tool.down({ x: 97, y: 76 });
    expect(tool.up({ x: 90, y: 70 })).toBe(false);
  });

  it("draws the thirds only while the corners still go around a shape", () => {
    const lines = (placed: StudioPerspective) => {
      const preview = new PerspectiveTool(placed, SIZE, vi.fn()).preview();
      return preview.kind === "corners" ? preview.lines : null;
    };

    expect(lines(CARD)).toHaveLength(4);
    expect(lines(corners([[96, 22], [30, 12], [100, 80], [18, 70]]))).toEqual([]);
  });
});

describe("the corners in the studio's state", () => {
  it("start at the picture's own, follow a drag, and start over with each picture", () => {
    let state = studioToolReducer({ ...initialToolState(), kind: "perspective" }, { type: "image-changed", width: 120, height: 90 });
    const moved: StudioPerspective[] = [];
    const tool = toolFor(state, null, (next) => moved.push(next));
    expect(tool).toBeInstanceOf(PerspectiveTool);

    tool!.down({ x: 1, y: 1 });
    tool!.up({ x: 31, y: 13 });
    state = studioToolReducer(state, { type: "set-perspective", corners: moved[0] });

    expect(state.perspective?.top_left).toEqual({ x: 30, y: 12 });
    // The next tool, made from the state, starts where the drag ended.
    expect(toolFor(state)?.preview()).toMatchObject({ corners: { top_left: { x: 30, y: 12 } } });
    state = studioToolReducer(state, { type: "image-changed", width: 60, height: 40 });
    expect(state.perspective).toBeNull();
  });

  it("keeps no step for Undo when a corner moves, since no selection changed", () => {
    const state = studioToolReducer({ ...initialToolState(), kind: "perspective" }, { type: "image-changed", width: 120, height: 90 });

    snapshotBeforeGesture(state);
    expect(state.history.canUndo).toBe(false);
    snapshotBeforeGesture({ ...state, kind: "brush" });
    expect(state.history.canUndo).toBe(true);
  });
});
