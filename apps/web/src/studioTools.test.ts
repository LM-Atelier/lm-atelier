import { describe, expect, it } from "vitest";
import { coverage, createMask, fillRect, isEmpty } from "./studioMasks";
import { BrushTool, LassoTool, RectTool } from "./studioTools";

describe("brush tool", () => {
  it("paints a continuous stroke across a drag and reports the change", () => {
    const mask = createMask(60, 20);
    const brush = new BrushTool(mask, 3);
    brush.down({ x: 5, y: 10 });
    brush.move({ x: 25, y: 10 });
    brush.move({ x: 45, y: 10 });
    expect(brush.up({ x: 55, y: 10 })).toBe(true);
    for (let x = 5; x <= 55; x += 5) {
      expect(mask.data[10 * 60 + x]).toBe(255);
    }
  });

  it("erases when constructed with a zero value", () => {
    const mask = createMask(20, 20);
    fillRect(mask, 0, 0, 20, 20);
    const eraser = new BrushTool(mask, 2, 0);
    eraser.down({ x: 10, y: 10 });
    eraser.up({ x: 10, y: 10 });
    expect(mask.data[10 * 20 + 10]).toBe(0);
    expect(mask.data[0]).toBe(255);
  });

  it("previews the cursor while hovering and while stroking", () => {
    const mask = createMask(10, 10);
    const brush = new BrushTool(mask, 4);
    expect(brush.preview().kind).toBe("none");
    brush.move({ x: 3, y: 3 });
    expect(brush.preview()).toEqual({
      kind: "brush-cursor",
      center: { x: 3, y: 3 },
      radius: 4,
    });
    // Hover alone must not paint.
    expect(isEmpty(mask)).toBe(true);
  });
});

describe("rect tool", () => {
  it("fills only on release and previews while dragging", () => {
    const mask = createMask(30, 30);
    const rect = new RectTool(mask);
    rect.down({ x: 5, y: 5 });
    rect.move({ x: 20, y: 18 });
    expect(rect.preview()).toEqual({
      kind: "rect",
      from: { x: 5, y: 5 },
      to: { x: 20, y: 18 },
    });
    expect(isEmpty(mask)).toBe(true);
    expect(rect.up({ x: 25, y: 25 })).toBe(true);
    expect(mask.data[10 * 30 + 10]).toBe(255);
    expect(rect.preview().kind).toBe("none");
  });

  it("treats a click without a drag as no selection", () => {
    const mask = createMask(10, 10);
    const rect = new RectTool(mask);
    rect.down({ x: 5, y: 5 });
    expect(rect.up({ x: 5.4, y: 5.2 })).toBe(false);
    expect(isEmpty(mask)).toBe(true);
  });
});

describe("lasso tool", () => {
  it("closes the path on release and fills its interior", () => {
    const mask = createMask(40, 40);
    const lasso = new LassoTool(mask);
    lasso.down({ x: 10, y: 10 });
    lasso.move({ x: 30, y: 10 });
    lasso.move({ x: 30, y: 30 });
    expect(lasso.up({ x: 10, y: 30 })).toBe(true);
    expect(mask.data[20 * 40 + 20]).toBe(255);
    expect(mask.data[5 * 40 + 5]).toBe(0);
  });

  it("drops sub-triangle gestures and dense duplicate points", () => {
    const mask = createMask(20, 20);
    const lasso = new LassoTool(mask, 2);
    lasso.down({ x: 5, y: 5 });
    lasso.move({ x: 5.5, y: 5.5 });
    expect(lasso.up({ x: 6, y: 6 })).toBe(false);
    expect(isEmpty(mask)).toBe(true);
    expect(coverage(mask)).toBe(0);
  });
});
