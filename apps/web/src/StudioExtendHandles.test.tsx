import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { StudioCanvas } from "./StudioCanvas";
import { StudioExtendHandles } from "./StudioExtendHandles";
import { initialToolState } from "./studioToolState";
import type { PointerTool } from "./studioTools";

/** Extend has no words in it: the frame is the whole instruction, so these
 * pin that moving an edge is what says how far to paint. */

function tools(margins?: Partial<Record<"top" | "right" | "bottom" | "left", number>>) {
  return {
    ...initialToolState(),
    kind: "extend" as const,
    margins: { top: 0, right: 0, bottom: 0, left: 0, ...margins },
  };
}

/** A 400 by 200 picture shown at twice its size, 50 pixels across and 20 down the canvas. */
const PICTURE = { width: 400, height: 200 };
const SHOWN = { x: 50, y: 20, width: 800, height: 400 };

/** jsdom's PointerEvent carries no client coordinates, so these are dispatched
 * as mouse events under the pointer event's name - which is what the browser
 * delivers anyway for a mouse. */
function pointer(type: string, x: number, y: number) {
  return new MouseEvent(type, { clientX: x, clientY: y, bubbles: true });
}

describe("StudioExtendHandles", () => {
  afterEach(cleanup);

  it("turns a drag outward into a share of the picture as it is shown", () => {
    const dispatch = vi.fn();
    const { container } = render(
      <StudioExtendHandles tools={tools()} dispatch={dispatch} picture={PICTURE} shown={SHOWN} />,
    );
    const right = container.querySelector('[data-side="right"]')!;

    fireEvent(right, pointer("pointerdown", 100, 50));
    window.dispatchEvent(pointer("pointermove", 300, 50));

    // 200 screen pixels across a picture shown 800 wide is a quarter of it,
    // though the picture itself is only 400 pixels wide.
    expect(dispatch).toHaveBeenCalledWith({ type: "set-margin", side: "right", fraction: 0.25 });
  });

  it("reads a drag on the top edge as upward rather than downward", () => {
    const dispatch = vi.fn();
    const { container } = render(
      <StudioExtendHandles tools={tools()} dispatch={dispatch} picture={PICTURE} shown={SHOWN} />,
    );
    const top = container.querySelector('[data-side="top"]')!;

    fireEvent(top, pointer("pointerdown", 50, 200));
    window.dispatchEvent(pointer("pointermove", 50, 100));

    expect(dispatch).toHaveBeenCalledWith({ type: "set-margin", side: "top", fraction: 0.25 });
  });

  it("keeps measuring a drag against the picture as it was shown when the drag began", () => {
    const dispatch = vi.fn();
    const { container, rerender } = render(
      <StudioExtendHandles tools={tools()} dispatch={dispatch} picture={PICTURE} shown={SHOWN} />,
    );

    fireEvent(container.querySelector('[data-side="left"]')!, pointer("pointerdown", 400, 50));
    // The view zooms out partway through the drag.
    rerender(
      <StudioExtendHandles
        tools={tools()}
        dispatch={dispatch}
        picture={PICTURE}
        shown={{ x: 250, y: 120, width: 400, height: 200 }}
      />,
    );
    window.dispatchEvent(pointer("pointermove", 200, 50));

    expect(dispatch).toHaveBeenLastCalledWith({ type: "set-margin", side: "left", fraction: 0.25 });
  });

  it("draws the new canvas around the picture where it is shown", () => {
    const { container } = render(
      <StudioExtendHandles
        tools={tools({ right: 0.25, top: 0.5 })}
        dispatch={vi.fn()}
        picture={PICTURE}
        shown={SHOWN}
      />,
    );
    const edge = container.querySelector(".studio-extend-edge")!;

    // A hundred pixels gained on the right and on top, each two on screen.
    expect(edge.getAttribute("x")).toBe("50");
    expect(edge.getAttribute("y")).toBe("-180");
    expect(edge.getAttribute("width")).toBe("1000");
    expect(edge.getAttribute("height")).toBe("600");
  });

  it("keeps a press or an arrow key on an edge from reaching the canvas beneath", () => {
    // A tool in the canvas's hand shows whether its handlers saw anything:
    // a press there starts a stroke and an arrow key moves the caret.
    const beneath: PointerTool = {
      appliesWhileMoving: false,
      down: vi.fn(),
      move: vi.fn(),
      up: vi.fn(() => false),
      cancel: vi.fn(),
      preview: () => ({ kind: "none" }),
    };
    const dispatch = vi.fn();
    const { container } = render(
      <StudioCanvas
        image={PICTURE as ImageBitmap}
        mask={null}
        tool={beneath}
        overlay={(shown) => (
          <StudioExtendHandles tools={tools()} dispatch={dispatch} picture={PICTURE} shown={shown} />
        )}
      />,
    );
    const bottom = container.querySelector('[data-side="bottom"]')!;

    fireEvent(bottom, pointer("pointerdown", 50, 200));
    fireEvent.keyDown(bottom, { key: "ArrowDown" });

    expect(beneath.down).not.toHaveBeenCalled();
    expect(beneath.move).not.toHaveBeenCalled();
    expect(dispatch).toHaveBeenCalledWith({
      type: "set-margin",
      side: "bottom",
      fraction: expect.closeTo(0.05, 5),
    });
  });

  it("moves an edge by keyboard, because a drag-only edge cannot be moved by everyone", () => {
    const dispatch = vi.fn();
    const { container } = render(
      <StudioExtendHandles tools={tools()} dispatch={dispatch} picture={PICTURE} shown={SHOWN} />,
    );

    fireEvent.keyDown(container.querySelector('[data-side="left"]')!, { key: "ArrowRight" });

    expect(dispatch).toHaveBeenCalledWith({
      type: "set-margin",
      side: "left",
      fraction: expect.closeTo(0.05, 5),
    });
  });

  it("says how far each edge already reaches", () => {
    render(
      <StudioExtendHandles
        tools={tools({ bottom: 0.4 })}
        dispatch={vi.fn()}
        picture={PICTURE}
        shown={SHOWN}
      />,
    );

    expect(screen.getByRole("button", { name: /Extend downward, currently 40 percent/ })).
      toBeInTheDocument();
  });

  it("ignores a move that was never a drag", () => {
    const dispatch = vi.fn();
    render(<StudioExtendHandles tools={tools()} dispatch={dispatch} picture={PICTURE} shown={SHOWN} />);

    window.dispatchEvent(pointer("pointermove", 300, 300));

    expect(dispatch).not.toHaveBeenCalled();
  });
});
