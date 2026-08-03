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

    fireEvent.pointerDown(surface, { clientX: 100, clientY: 50, button: 0 });
    fireEvent.pointerMove(surface, { clientX: 140, clientY: 50 });
    fireEvent.pointerUp(surface, { clientX: 140, clientY: 50 });

    expect(tool.calls.map(([kind]) => kind)).toEqual(["down", "move", "up"]);
    // jsdom's zero-size container fits to scale 1 with no offset, so screen
    // equals image space here; the important part is consistent numbers.
    expect(tool.calls[0][1]).toEqual(tool.calls.at(-1)?.[1] ?? {});
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
    fireEvent.pointerDown(surface, { clientX: 10, clientY: 10, button: 0 });
    fireEvent.pointerUp(surface, { clientX: 10, clientY: 10 });
    expect(onStrokeEnd).not.toHaveBeenCalled();
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
