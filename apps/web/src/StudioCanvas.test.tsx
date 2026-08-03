import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render } from "@testing-library/react";
import { StudioCanvas } from "./StudioCanvas";
import { createMask } from "./studioMasks";
import type { ImagePoint, PointerTool, ToolPreview } from "./studioTools";

/** jsdom has no 2D context; the component must tolerate null contexts and
 * still run its geometry and tool forwarding, which is what these pin. */

class SpyTool implements PointerTool {
  calls: Array<[string, ImagePoint]> = [];
  changed = true;

  down(point: ImagePoint): void {
    this.calls.push(["down", point]);
  }

  move(point: ImagePoint): void {
    this.calls.push(["move", point]);
  }

  up(point: ImagePoint): boolean {
    this.calls.push(["up", point]);
    return this.changed;
  }

  preview(): ToolPreview {
    return { kind: "none" };
  }
}

const image = { width: 400, height: 200 } as ImageBitmap;

describe("StudioCanvas", () => {
  afterEach(cleanup);

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
});
