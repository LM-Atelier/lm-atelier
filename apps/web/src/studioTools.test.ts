import { describe, expect, it } from "vitest";
import { coverage, createMask, fillRect, isEmpty } from "./studioMasks";
import { BrushTool, LassoTool, RectTool } from "./studioTools";
import { defaultInstruction, initialToolState, toolUsesMask } from "./studioToolState";

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

describe("abandoning a gesture", () => {
  it("leaves a rectangle and a lasso with nothing applied", () => {
    // These apply on close, so cancelling must not go through up() - that
    // is the call that commits, and cancelling through it would paint the
    // very thing being cancelled.
    const rectMask = createMask(40, 40);
    const rect = new RectTool(rectMask);
    rect.down({ x: 4, y: 4 });
    rect.move({ x: 30, y: 30 });
    rect.cancel();
    expect(coverage(rectMask)).toBe(0);
    expect(rect.up({ x: 30, y: 30 })).toBe(false);

    const lassoMask = createMask(40, 40);
    const lasso = new LassoTool(lassoMask);
    lasso.down({ x: 4, y: 4 });
    lasso.move({ x: 30, y: 6 });
    lasso.move({ x: 20, y: 30 });
    lasso.cancel();
    expect(coverage(lassoMask)).toBe(0);
  });

  it("keeps what a brush already painted, since undo is the way back", () => {
    const mask = createMask(40, 40);
    const brush = new BrushTool(mask, 5);
    brush.down({ x: 20, y: 20 });
    const painted = coverage(mask);
    expect(painted).toBeGreaterThan(0);

    brush.cancel();
    // A brush applies as it travels; there is nothing held back to discard,
    // and the snapshot taken at gesture start is what restores it.
    expect(coverage(mask)).toBe(painted);
    expect(brush.up({ x: 20, y: 20 })).toBe(false);
  });
});

describe("what travels with a turn", () => {
  it("keeps a drawn selection away from the tools that never asked for one", () => {
    // Gating on "not the instruct tool" sent a mask left over from the brush
    // with an Enhance or an Extend, silently narrowing work that is meant to
    // be about the whole picture.
    expect(toolUsesMask("brush")).toBe(true);
    expect(toolUsesMask("eraser")).toBe(true);
    expect(toolUsesMask("rect")).toBe(true);
    expect(toolUsesMask("lasso")).toBe(true);
    expect(toolUsesMask("enhance")).toBe(false);
    expect(toolUsesMask("extend")).toBe(false);
    expect(toolUsesMask("instruct")).toBe(false);
  });

  it("gives the wordless tools something true to say", () => {
    // The turn contract requires text and these two ask for none, so an empty
    // box was refused by the server before anything ran.
    const enhance = { ...initialToolState(), kind: "enhance" as const, upscaleFactor: 2 };
    expect(defaultInstruction(enhance)).toBe("Enhance to 2x");

    const extend = {
      ...initialToolState(),
      kind: "extend" as const,
      margins: { top: 64, right: 0, bottom: 0, left: 32 },
    };
    expect(defaultInstruction(extend)).toBe("Extend past the top, left");

    expect(defaultInstruction({ ...initialToolState(), kind: "instruct" as const })).toBe("");
  });
});
