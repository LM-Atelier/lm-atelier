import { useCallback, useEffect, useRef, useState } from "react";
import type { StudioToolState, StudioToolAction } from "./studioToolState";

type Side = "top" | "right" | "bottom" | "left";

const SIDES: Array<{ side: Side; label: string; axis: "x" | "y"; sign: 1 | -1 }> = [
  { side: "top", label: "Extend upward", axis: "y", sign: -1 },
  { side: "right", label: "Extend to the right", axis: "x", sign: 1 },
  { side: "bottom", label: "Extend downward", axis: "y", sign: 1 },
  { side: "left", label: "Extend to the left", axis: "x", sign: -1 },
];

/** How far one keyboard press extends a side, as a fraction of the picture. */
const KEYBOARD_STEP = 0.05;

/** A synthetic or touch-only pointer event can omit client coordinates, and a
 * missing one must not become a NaN margin that silently disables Extend. */
function coordinate(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

/** The four edges of the picture, draggable outward.
 *
 * Extend is the tool with no words in it: what the workflow paints is decided
 * entirely by where the frame ends up, so the frame is the control. Dragging
 * an edge outward is the whole instruction.
 *
 * Margins are fractions of the picture rather than pixels, because the drag
 * happens on a view at some zoom and a count of screen pixels means nothing
 * to a workflow that never saw the screen.
 */
export function StudioExtendHandles({
  tools,
  dispatch,
  size,
}: {
  tools: StudioToolState;
  dispatch: (action: StudioToolAction) => void;
  /** The picture's on-screen size, which is what a drag is measured against. */
  size: { width: number; height: number };
}) {
  const dragging = useRef<{ side: Side; from: number; start: number } | null>(null);
  const [live, setLive] = useState<Side | null>(null);

  const onMove = useCallback(
    (event: PointerEvent) => {
      const drag = dragging.current;
      if (!drag) return;
      const config = SIDES.find((item) => item.side === drag.side)!;
      const along = coordinate(config.axis === "x" ? event.clientX : event.clientY);
      const span = config.axis === "x" ? size.width : size.height;
      if (span <= 0) return;
      const moved = ((along - drag.from) * config.sign) / span;
      dispatch({ type: "set-margin", side: drag.side, fraction: drag.start + moved });
    },
    [dispatch, size.height, size.width],
  );

  const onUp = useCallback(() => {
    dragging.current = null;
    setLive(null);
  }, []);

  useEffect(() => {
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [onMove, onUp]);

  return (
    <div className="studio-extend-handles">
      {SIDES.map(({ side, label, axis }) => (
        <button
          key={side}
          type="button"
          className={`studio-extend-handle ${side} ${live === side ? "dragging" : ""}`}
          data-side={side}
          aria-label={`${label}, currently ${Math.round(tools.margins[side] * 100)} percent`}
          style={{ [axis === "x" ? "width" : "height"]: `${8 + tools.margins[side] * 100}px` }}
          onPointerDown={(event) => {
            event.preventDefault();
            (event.target as Element).setPointerCapture?.(event.pointerId);
            dragging.current = {
              side,
              from: coordinate(axis === "x" ? event.clientX : event.clientY),
              start: tools.margins[side],
            };
            setLive(side);
          }}
          onKeyDown={(event) => {
            // The same control by keyboard: an edge that can only be dragged
            // is an edge some people cannot move at all.
            const grow = event.key === "ArrowRight" || event.key === "ArrowDown";
            const shrink = event.key === "ArrowLeft" || event.key === "ArrowUp";
            if (!grow && !shrink) return;
            event.preventDefault();
            dispatch({
              type: "set-margin",
              side,
              fraction: tools.margins[side] + (grow ? KEYBOARD_STEP : -KEYBOARD_STEP),
            });
          }}
        />
      ))}
    </div>
  );
}
