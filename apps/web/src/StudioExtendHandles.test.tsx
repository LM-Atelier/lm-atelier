import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { StudioExtendHandles } from "./StudioExtendHandles";
import { initialToolState } from "./studioToolState";

/** Extend has no words in it: the frame is the whole instruction, so these
 * pin that moving an edge is what says how far to paint. */

function tools(margins?: Partial<Record<"top" | "right" | "bottom" | "left", number>>) {
  return {
    ...initialToolState(),
    kind: "extend" as const,
    margins: { top: 0, right: 0, bottom: 0, left: 0, ...margins },
  };
}

const SIZE = { width: 400, height: 200 };

/** jsdom's PointerEvent carries no client coordinates, so these are dispatched
 * as mouse events under the pointer event's name - which is what the browser
 * delivers anyway for a mouse. */
function pointer(type: string, x: number, y: number) {
  return new MouseEvent(type, { clientX: x, clientY: y, bubbles: true });
}

describe("StudioExtendHandles", () => {
  afterEach(cleanup);

  it("turns a drag outward into a fraction of the picture", () => {
    const dispatch = vi.fn();
    const { container } = render(
      <StudioExtendHandles tools={tools()} dispatch={dispatch} size={SIZE} />,
    );
    const right = container.querySelector('[data-side="right"]')!;

    fireEvent(right, pointer("pointerdown", 100, 50));
    window.dispatchEvent(pointer("pointermove", 200, 50));

    // 100px across a 400px picture is a quarter of it, whatever the zoom was.
    expect(dispatch).toHaveBeenCalledWith({ type: "set-margin", side: "right", fraction: 0.25 });
  });

  it("reads a drag on the top edge as upward rather than downward", () => {
    const dispatch = vi.fn();
    const { container } = render(
      <StudioExtendHandles tools={tools()} dispatch={dispatch} size={SIZE} />,
    );
    const top = container.querySelector('[data-side="top"]')!;

    fireEvent(top, pointer("pointerdown", 50, 100));
    window.dispatchEvent(pointer("pointermove", 50, 50));

    expect(dispatch).toHaveBeenCalledWith({ type: "set-margin", side: "top", fraction: 0.25 });
  });

  it("moves an edge by keyboard, because a drag-only edge cannot be moved by everyone", () => {
    const dispatch = vi.fn();
    const { container } = render(
      <StudioExtendHandles tools={tools()} dispatch={dispatch} size={SIZE} />,
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
      <StudioExtendHandles tools={tools({ bottom: 0.4 })} dispatch={vi.fn()} size={SIZE} />,
    );

    expect(screen.getByRole("button", { name: /Extend downward, currently 40 percent/ })).
      toBeInTheDocument();
  });

  it("ignores a move that was never a drag", () => {
    const dispatch = vi.fn();
    render(<StudioExtendHandles tools={tools()} dispatch={dispatch} size={SIZE} />);

    window.dispatchEvent(pointer("pointermove", 300, 300));

    expect(dispatch).not.toHaveBeenCalled();
  });
});
