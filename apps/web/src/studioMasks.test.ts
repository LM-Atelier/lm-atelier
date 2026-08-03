import { describe, expect, it } from "vitest";
import {
  cloneMask,
  coverage,
  createMask,
  decodeMask,
  DEFAULT_MASK_HISTORY_BYTES,
  encodeMask,
  feather,
  fillPolygon,
  fillRect,
  invert,
  isEmpty,
  MaskHistory,
  stampCircle,
  strokeSegment,
  toAlphaImageData,
} from "./studioMasks";

describe("mask rasters", () => {
  it("starts empty and bounds its dimensions", () => {
    const mask = createMask(64, 32);
    expect(isEmpty(mask)).toBe(true);
    expect(coverage(mask)).toBe(0);
    expect(() => createMask(0, 10)).toThrow(/dimensions/);
    expect(() => createMask(10_000, 10)).toThrow(/dimensions/);
    expect(() => createMask(10.5, 10)).toThrow(/dimensions/);
  });

  it("stamps a circle that covers its center and respects edges", () => {
    const mask = createMask(20, 20);
    stampCircle(mask, 10, 10, 4);
    expect(mask.data[10 * 20 + 10]).toBe(255);
    expect(mask.data[0]).toBe(0);
    // Stamping at a corner clips instead of wrapping.
    stampCircle(mask, 0, 0, 3);
    expect(mask.data[0]).toBe(255);
    expect(mask.data[19]).toBe(0);
    expect(isEmpty(mask)).toBe(false);
  });

  it("strokes a continuous segment with no gaps at brush spacing", () => {
    const mask = createMask(100, 20);
    strokeSegment(mask, { x: 10, y: 10 }, { x: 90, y: 10 }, 4);
    for (let x = 10; x <= 90; x += 1) {
      expect(mask.data[10 * 100 + x]).toBe(255);
    }
  });

  it("erases with a zero-value stroke", () => {
    const mask = createMask(40, 40);
    fillRect(mask, 0, 0, 40, 40);
    strokeSegment(mask, { x: 20, y: 0 }, { x: 20, y: 40 }, 3, 0);
    expect(mask.data[20 * 40 + 20]).toBe(0);
    expect(mask.data[20 * 40 + 2]).toBe(255);
  });

  it("fills rectangles with exclusive pixel edges", () => {
    const mask = createMask(10, 10);
    fillRect(mask, 2, 2, 5, 5);
    expect(mask.data[2 * 10 + 2]).toBe(255);
    expect(mask.data[4 * 10 + 4]).toBe(255);
    expect(mask.data[5 * 10 + 5]).toBe(0);
    expect(coverage(mask)).toBeCloseTo(9 / 100);
  });

  it("fills a lasso polygon by even-odd scanlines", () => {
    const mask = createMask(20, 20);
    fillPolygon(mask, [
      { x: 5, y: 5 },
      { x: 15, y: 5 },
      { x: 15, y: 15 },
      { x: 5, y: 15 },
    ]);
    expect(mask.data[10 * 20 + 10]).toBe(255);
    expect(mask.data[2 * 20 + 2]).toBe(0);
    // Degenerate polygons do nothing rather than throwing.
    const untouched = createMask(8, 8);
    fillPolygon(untouched, [{ x: 1, y: 1 }, { x: 2, y: 2 }]);
    expect(isEmpty(untouched)).toBe(true);
  });

  it("inverts coverage exactly", () => {
    const mask = createMask(4, 4);
    fillRect(mask, 0, 0, 2, 4);
    invert(mask);
    expect(mask.data[0]).toBe(0);
    expect(mask.data[3]).toBe(255);
    invert(mask);
    expect(mask.data[0]).toBe(255);
  });

  it("feathers edges into a soft ramp without moving the core", () => {
    const mask = createMask(41, 41);
    fillRect(mask, 8, 8, 33, 33);
    feather(mask, 3);
    // Deep inside (beyond two blur radii from any edge) stays fully
    // selected; well outside stays untouched.
    expect(mask.data[20 * 41 + 20]).toBe(255);
    expect(mask.data[1 * 41 + 1]).toBe(0);
    // The former hard edge is now intermediate.
    const edge = mask.data[20 * 41 + 33];
    expect(edge).toBeGreaterThan(0);
    expect(edge).toBeLessThan(255);
    // Zero radius is a no-op.
    const untouched = createMask(8, 8);
    fillRect(untouched, 2, 2, 6, 6);
    const before = [...untouched.data];
    feather(untouched, 0);
    expect([...untouched.data]).toEqual(before);
  });

  it("exports RGBA with the mask as alpha", () => {
    const mask = createMask(2, 1);
    mask.data[1] = 128;
    const rgba = toAlphaImageData(mask, [10, 20, 30]);
    expect([...rgba]).toEqual([10, 20, 30, 0, 10, 20, 30, 128]);
  });
});

describe("mask history", () => {
  it("round-trips any raster through run-length encoding", () => {
    const mask = createMask(37, 11);
    fillRect(mask, 3, 2, 20, 9);
    stampCircle(mask, 30, 5, 4, 128);
    const restored = decodeMask(encodeMask(mask));
    expect([...restored.data]).toEqual([...mask.data]);
    expect(restored.width).toBe(37);

    // An all-zero mask and an all-set mask are the degenerate ends.
    const empty = createMask(8, 8);
    expect(isEmpty(decodeMask(encodeMask(empty)))).toBe(true);
    const full = createMask(8, 8);
    fillRect(full, 0, 0, 8, 8);
    expect(coverage(decodeMask(encodeMask(full)))).toBe(1);
  });

  it("bounds memory by bytes and evicts the oldest steps first", () => {
    const mask = createMask(64, 64);
    const stripe = (step: number) =>
      fillRect(mask, step * 2, 0, step * 2 + 1, 64, step % 2 === 0 ? 255 : 0);

    // Measure one snapshot, then budget for roughly three.
    const probe = new MaskHistory();
    stripe(0);
    probe.push(mask);
    const perSnapshot = probe.usedBytes;
    expect(perSnapshot).toBeGreaterThan(0);

    const budget = perSnapshot * 3;
    const history = new MaskHistory(64, budget);
    const unbounded = new MaskHistory(64, Number.MAX_SAFE_INTEGER);
    for (let step = 1; step < 12; step += 1) {
      history.push(mask);
      unbounded.push(mask);
      stripe(step);
    }

    // Later snapshots are larger than the first, so the honest bound is "a
    // small multiple of budget" - and far below keeping every step.
    expect(history.usedBytes).toBeLessThan(budget * 4);
    expect(history.usedBytes).toBeLessThan(unbounded.usedBytes / 2);
    // Eviction takes the oldest steps, never the newest: undo still works.
    expect(history.canUndo).toBe(true);
    expect(history.undo(mask)).not.toBeNull();
  });

  it("always keeps at least one undo step, however small the budget", () => {
    // A budget below one snapshot cannot be honored without discarding the
    // user's only way back; the last step survives deliberately.
    const history = new MaskHistory(64, 1);
    const mask = createMask(32, 32);
    fillRect(mask, 0, 0, 16, 32);
    history.push(mask);
    fillRect(mask, 16, 0, 32, 32, 128);

    expect(history.canUndo).toBe(true);
    const undone = history.undo(mask)!;
    expect(coverage(undone)).toBeCloseTo(0.5);
  });

  it("drops redo before sacrificing undo history", () => {
    const mask = createMask(48, 48);
    const probe = new MaskHistory();
    fillRect(mask, 0, 0, 24, 48);
    probe.push(mask);

    const history = new MaskHistory(64, probe.usedBytes * 2);
    history.push(mask);
    fillRect(mask, 24, 0, 48, 48, 128);
    const undone = history.undo(mask)!;
    expect(history.canRedo).toBe(true);
    // Pushing new work clears redo outright, as a new branch must.
    history.push(undone);
    expect(history.canRedo).toBe(false);
  });

  it("undoes to the pushed state and redoes forward", () => {
    const history = new MaskHistory();
    const mask = createMask(4, 4);
    history.push(mask);
    fillRect(mask, 0, 0, 4, 4);

    const undone = history.undo(mask);
    expect(undone).not.toBeNull();
    expect(isEmpty(undone!)).toBe(true);
    expect(history.canRedo).toBe(true);

    const redone = history.redo(undone!);
    expect(redone).not.toBeNull();
    expect(coverage(redone!)).toBe(1);
  });

  it("caps depth and clears redo on a new stroke", () => {
    const history = new MaskHistory(2, DEFAULT_MASK_HISTORY_BYTES);
    const mask = createMask(2, 2);
    history.push(mask);
    history.push(mask);
    history.push(mask);
    expect(history.canUndo).toBe(true);
    let current = cloneMask(mask);
    current = history.undo(current)!;
    current = history.undo(current)!;
    // Depth 2: the third pushed state was evicted.
    expect(history.undo(current)).toBeNull();

    history.push(current);
    expect(history.canRedo).toBe(false);
  });
});
