import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render } from "@testing-library/react";
import { StudioCanvas } from "./StudioCanvas";
import { createMask } from "./studioMasks";
import type { ImagePoint, PointerTool, ToolPreview } from "./studioTools";

/** jsdom has no 2D context; the component must tolerate null contexts and
 * still run its geometry and tool forwarding, which is what these pin. */

class SpyTool implements PointerTool {
  calls: Array<[string, ImagePoint]> = [];
  viewportScales: number[] = [];
  changed = true;
  appliesWhileMoving = false;

  down(point: ImagePoint, viewportScale = 1): void {
    this.calls.push(["down", point]);
    this.viewportScales.push(viewportScale);
  }

  move(point: ImagePoint, viewportScale = 1): void {
    this.calls.push(["move", point]);
    this.viewportScales.push(viewportScale);
  }

  up(point: ImagePoint, viewportScale = 1): boolean {
    this.calls.push(["up", point]);
    this.viewportScales.push(viewportScale);
    return this.changed;
  }

  cancel(): void {
    this.calls.push(["cancel", { x: 0, y: 0 }]);
  }

  preview(): ToolPreview {
    return { kind: "none" };
  }
}

const image = { width: 400, height: 200 } as ImageBitmap;

describe("StudioCanvas", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders the three layers sized to the image", () => {
    const { container } = render(
      <StudioCanvas image={image} mask={createMask(400, 200)} tool={null} />,
    );
    const canvases = container.querySelectorAll("canvas");
    expect(canvases).toHaveLength(3);
    expect(canvases[0]).toHaveAttribute("width", "400");
    expect(canvases[2]).toHaveAttribute("data-layer", "interaction");
  });

  it("forwards unprojected pointer gestures to the tool and reports stroke end", () => {
    const tool = new SpyTool();
    const onStrokeEnd = vi.fn();
    const { container } = render(
      <StudioCanvas image={image} mask={null} tool={tool} onStrokeEnd={onStrokeEnd} />,
    );
    const surface = container.querySelector(".studio-canvas")!;

    fireEvent.pointerDown(surface, { pointerType: "mouse", clientX: 100, clientY: 50, button: 0 });
    fireEvent.pointerMove(surface, { pointerType: "mouse", clientX: 140, clientY: 50 });
    fireEvent.pointerUp(surface, { pointerType: "mouse", clientX: 140, clientY: 50 });

    expect(tool.calls.map(([kind]) => kind)).toEqual(["down", "move", "up"]);
    // jsdom does not carry client coordinates on synthetic PointerEvents, so
    // this asserts what it can express: finite, non-NaN points reach the tool.
    // The real screen-to-image math is proven exhaustively in
    // studioViewport.test.ts, which needs no DOM.
    for (const [, point] of tool.calls) {
      expect(Number.isFinite(point.x)).toBe(true);
      expect(Number.isFinite(point.y)).toBe(true);
    }
    expect(onStrokeEnd).toHaveBeenCalledTimes(1);
  });

  it("suppresses the stroke-end callback when the gesture changed nothing", () => {
    const tool = new SpyTool();
    tool.changed = false;
    const onStrokeEnd = vi.fn();
    const { container } = render(
      <StudioCanvas image={image} mask={null} tool={tool} onStrokeEnd={onStrokeEnd} />,
    );
    const surface = container.querySelector(".studio-canvas")!;
    fireEvent.pointerDown(surface, { pointerType: "mouse", clientX: 10, clientY: 10, button: 0 });
    fireEvent.pointerUp(surface, { pointerType: "mouse", clientX: 10, clientY: 10 });
    expect(onStrokeEnd).not.toHaveBeenCalled();
  });


  it("snapshots before the first mutation, not after the stroke", () => {
    const tool = new SpyTool();
    const order: string[] = [];
    const { container } = render(
      <StudioCanvas
        image={image}
        mask={null}
        tool={tool}
        onGestureStart={() => order.push("snapshot")}
        onStrokeEnd={() => order.push("stroke-end")}
      />,
    );
    const surface = container.querySelector(".studio-canvas")!;

    fireEvent.pointerDown(surface, { pointerType: "mouse", clientX: 20, clientY: 20, button: 0, pointerId: 1 });
    fireEvent.pointerUp(surface, { pointerType: "mouse", clientX: 30, clientY: 20, pointerId: 1 });

    // The snapshot precedes the tool's first mutation; snapshotting at stroke
    // end would capture the already-painted raster and make Undo a no-op.
    expect(order).toEqual(["snapshot", "stroke-end"]);
    expect(tool.calls[0][0]).toBe("down");
  });

  it("ends the stroke when a second pointer arrives and pans instead", () => {
    const tool = new SpyTool();
    const { container } = render(<StudioCanvas image={image} mask={null} tool={tool} />);
    const surface = container.querySelector(".studio-canvas")!;

    fireEvent.pointerDown(surface, { pointerType: "mouse", clientX: 10, clientY: 10, button: 0, pointerId: 1 });
    fireEvent.pointerDown(surface, { pointerType: "mouse", clientX: 60, clientY: 60, button: 0, pointerId: 2 });

    // The in-progress stroke is closed; the second pointer never starts one.
    expect(tool.calls.map(([kind]) => kind)).toEqual(["down", "up"]);

    fireEvent.pointerMove(surface, { pointerType: "mouse", clientX: 80, clientY: 60, pointerId: 2 });
    expect(tool.calls.map(([kind]) => kind)).toEqual(["down", "up"]);
  });

  it("closes a cancelled gesture and clears pan state on lost capture", () => {
    const tool = new SpyTool();
    const onStrokeEnd = vi.fn();
    const { container } = render(
      <StudioCanvas image={image} mask={null} tool={tool} onStrokeEnd={onStrokeEnd} />,
    );
    const surface = container.querySelector(".studio-canvas")!;

    fireEvent.pointerDown(surface, { pointerType: "mouse", clientX: 10, clientY: 10, button: 0, pointerId: 3 });
    fireEvent.pointerCancel(surface, { pointerType: "mouse", clientX: 15, clientY: 10, pointerId: 3 });

    expect(tool.calls.map(([kind]) => kind)).toEqual(["down", "up"]);
    expect(onStrokeEnd).toHaveBeenCalledTimes(1);

    // A further move must not continue the cancelled stroke.
    fireEvent.pointerMove(surface, { pointerType: "mouse", clientX: 40, clientY: 10, pointerId: 3 });
    expect(tool.calls.filter(([kind]) => kind === "move")).toHaveLength(1);
  });

  it("zooms the layer transform about the wheel cursor", () => {
    const { container } = render(<StudioCanvas image={image} mask={null} tool={null} />);
    const surface = container.querySelector(".studio-canvas")!;
    const layers = container.querySelector<HTMLElement>(".studio-canvas-layers")!;
    const before = layers.style.transform;

    fireEvent.wheel(surface, { deltaY: -120, clientX: 200, clientY: 100 });

    expect(layers.style.transform).not.toBe(before);
    expect(layers.style.transform).toContain("scale(1.2");
  });

  it("passes the live viewport scale to drawing tools", () => {
    const tool = new SpyTool();
    const { container } = render(<StudioCanvas image={image} mask={null} tool={tool} />);
    const surface = container.querySelector(".studio-canvas")!;

    fireEvent.wheel(surface, { deltaY: -120, clientX: 200, clientY: 100 });
    fireEvent.pointerDown(surface, {
      pointerType: "mouse",
      clientX: 100,
      clientY: 50,
      button: 0,
      pointerId: 1,
    });
    fireEvent.pointerUp(surface, {
      pointerType: "mouse",
      clientX: 100,
      clientY: 50,
      pointerId: 1,
    });

    expect(tool.viewportScales).toEqual([1.2, 1.2]);
  });

  it("is reachable and steerable from the keyboard", () => {
    const { container } = render(
      <StudioCanvas image={image} mask={createMask(400, 200)} tool={null} />,
    );
    const surface = container.querySelector(".studio-canvas") as HTMLElement;
    const layers = () => container.querySelector(".studio-canvas-layers") as HTMLElement;

    // role="application" hands every key to the element instead of to the
    // screen reader. That was a lie while the element could take neither
    // focus nor a keystroke, so what this pins is that both are now true.
    expect(surface).toHaveAttribute("role", "application");
    expect(surface).toHaveAttribute("tabindex", "0");

    const start = layers().style.transform;
    fireEvent.keyDown(surface, { key: "ArrowRight" });
    const panned = layers().style.transform;
    expect(panned).not.toBe(start);

    fireEvent.keyDown(surface, { key: "+" });
    expect(layers().style.transform).not.toBe(panned);
  });


  it("lets a selection be made without a pointer at all", () => {
    const tool = new SpyTool();
    const onGestureStart = vi.fn();
    const onStrokeEnd = vi.fn();
    const { container } = render(
      <StudioCanvas
        image={image}
        mask={createMask(400, 200)}
        tool={tool}
        onGestureStart={onGestureStart}
        onStrokeEnd={onStrokeEnd}
      />,
    );
    const surface = container.querySelector(".studio-canvas") as HTMLElement;

    // Every selection tool was pointer-only, so the whole selection-based
    // half of the studio was unreachable by keyboard. The caret drives the
    // same tools the pointer does.
    fireEvent.keyDown(surface, { key: "Enter" });
    expect(onGestureStart).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(surface, { key: "ArrowRight" });
    fireEvent.keyDown(surface, { key: "ArrowDown" });
    fireEvent.keyDown(surface, { key: "Enter" });

    expect(tool.calls.map(([kind]) => kind)).toEqual(["down", "move", "move", "up"]);
    // The caret starts at the middle of the picture and moves in image
    // coordinates, not screen ones.
    expect(tool.calls[0][1]).toEqual({ x: 200, y: 100 });
    expect(tool.calls[3][1]).toEqual({ x: 220, y: 120 });
    expect(onStrokeEnd).toHaveBeenCalledTimes(1);
  });

  it("keeps panning available while a tool is in hand", () => {
    const tool = new SpyTool();
    const { container } = render(
      <StudioCanvas image={image} mask={createMask(400, 200)} tool={tool} />,
    );
    const surface = container.querySelector(".studio-canvas") as HTMLElement;
    const layers = () => container.querySelector(".studio-canvas-layers") as HTMLElement;

    const before = layers().style.transform;
    fireEvent.keyDown(surface, { key: "ArrowRight", altKey: true });
    // Alt pans; without it the same key would have moved the caret, which
    // is what a tool being selected should mean.
    expect(layers().style.transform).not.toBe(before);
    expect(tool.calls).toEqual([]);
  });

  it("abandons a keyboard selection on Escape", () => {
    const tool = new SpyTool();
    const onStrokeEnd = vi.fn();
    const { container } = render(
      <StudioCanvas
        image={image}
        mask={createMask(400, 200)}
        tool={tool}
        onStrokeEnd={onStrokeEnd}
      />,
    );
    const surface = container.querySelector(".studio-canvas") as HTMLElement;

    fireEvent.keyDown(surface, { key: "Enter" });
    fireEvent.keyDown(surface, { key: "Escape" });
    fireEvent.keyDown(surface, { key: "ArrowRight" });

    // The stroke is closed, so the arrow that follows moves the caret
    // rather than silently extending an abandoned selection.
    expect(onStrokeEnd).not.toHaveBeenCalled();
    expect(tool.calls.map(([kind]) => kind)).toEqual(["down", "cancel", "move"]);
  });


  /** jsdom has neither a 2D context nor ImageData, so the tint is observed
   * through the paint call rather than through pixels. */
  function recordPaints() {
    const painted: string[] = [];
    const context = new Proxy(
      { putImageData: () => painted.push("mask") },
      {
        get: (target: Record<string, unknown>, key: string) =>
          key in target ? target[key] : () => undefined,
        set: () => true,
      },
    );
    vi.stubGlobal("ImageData", class {});
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      context as unknown as CanvasRenderingContext2D,
    );
    return painted;
  }

  function strokeAcross(tool: SpyTool) {
    const { container } = render(
      <StudioCanvas image={image} mask={createMask(400, 200)} tool={tool} />,
    );
    const surface = container.querySelector(".studio-canvas")!;
    return () => {
      fireEvent.pointerDown(surface, { pointerType: "mouse", clientX: 10, clientY: 10, button: 0 });
      fireEvent.pointerMove(surface, { pointerType: "mouse", clientX: 40, clientY: 30 });
    };
  }

  it("paints the selection as the brush travels, not when it stops", () => {
    // The defect: a brush writes straight into the raster and the version
    // only bumps at stroke end, so the tint arrived after the pointer lifted
    // and drawing felt like guessing.
    const painted = recordPaints();
    const tool = new SpyTool();
    tool.appliesWhileMoving = true;
    const stroke = strokeAcross(tool);
    painted.length = 0;

    stroke();

    expect(painted.length).toBeGreaterThanOrEqual(2);
  });

  it("leaves the tint alone for a gesture that commits nothing until it closes", () => {
    const painted = recordPaints();
    const stroke = strokeAcross(new SpyTool());
    painted.length = 0;

    stroke();

    expect(painted).toHaveLength(0);
  });
});
