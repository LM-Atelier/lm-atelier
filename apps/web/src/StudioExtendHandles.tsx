import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { extendedFrame, type ExtendSide } from "./studioExtend";
import type { StudioToolState, StudioToolAction } from "./studioToolState";
import type { ScreenRect } from "./studioViewport";

const SIDES: Array<{ side: ExtendSide; label: string; axis: "x" | "y"; sign: 1 | -1 }> = [
  { side: "top", label: "Extend upward", axis: "y", sign: -1 },
  { side: "right", label: "Extend to the right", axis: "x", sign: 1 },
  { side: "bottom", label: "Extend downward", axis: "y", sign: 1 },
  { side: "left", label: "Extend to the left", axis: "x", sign: -1 },
];

/** How far one keyboard press extends a side, as a fraction of the picture. */
const KEYBOARD_STEP = 0.05;

/** A grip's thickness on screen, and the furthest it runs along its edge, at any zoom. */
const GRIP = 12;
const GRIP_REACH = 160;

/** A synthetic or touch-only pointer event can omit client coordinates, and a
 * missing one must not become a NaN margin that silently disables Extend. */
function coordinate(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

/** A position that keeps something this long inside the canvas, however far its edge has gone.
 *
 * An edge dragged out past the view would otherwise take its grip with it, and
 * a grip nobody can reach is an edge nobody can bring back.
 */
function inView(position: number, length: number): string {
  return `clamp(0px, ${position}px, calc(100% - ${length}px))`;
}

/** Where a side's grip sits: across the middle of that edge of the new canvas. */
function gripStyle(side: ExtendSide, frame: ScreenRect): CSSProperties {
  const across = side === "top" || side === "bottom";
  const length = Math.max(2 * GRIP, Math.min(GRIP_REACH, (across ? frame.width : frame.height) / 2));
  if (across) {
    const edge = side === "top" ? frame.y : frame.y + frame.height;
    return {
      left: inView(frame.x + (frame.width - length) / 2, length),
      top: inView(edge - GRIP / 2, GRIP),
      width: length,
      height: GRIP,
    };
  }
  const edge = side === "left" ? frame.x : frame.x + frame.width;
  return {
    left: inView(edge - GRIP / 2, GRIP),
    top: inView(frame.y + (frame.height - length) / 2, length),
    width: GRIP,
    height: length,
  };
}

/** A rectangle as a closed path, for cutting the picture out of the new canvas. */
function outline(rect: ScreenRect): string {
  return `M${rect.x} ${rect.y}h${rect.width}v${rect.height}h${-rect.width}z`;
}

/** The canvas the extension makes, drawn around the picture, its four edges draggable outward.
 *
 * Extend is the tool with no words in it: what the workflow paints is decided
 * entirely by where the frame ends up, so the frame is the control. Dragging
 * an edge outward is the whole instruction.
 *
 * The frame is drawn where the canvas shows the picture, at the zoom it shows
 * it at, and a drag is divided by the picture's size on screen. The same share
 * of the picture then takes the same stretch of the frame however far in or out
 * the view is, and the margin stays a share of the picture: a count of screen
 * pixels means nothing to a workflow that never saw the screen.
 */
export function StudioExtendHandles({
  tools,
  dispatch,
  picture,
  shown,
}: {
  tools: StudioToolState;
  dispatch: (action: StudioToolAction) => void;
  /** The picture's size in its own pixels, which the margins are shares of. */
  picture: { width: number; height: number };
  /** Where the canvas shows the picture now, in its CSS pixels. */
  shown: ScreenRect;
}) {
  const dragging = useRef<{ side: ExtendSide; from: number; start: number; span: number } | null>(
    null,
  );
  const [live, setLive] = useState<ExtendSide | null>(null);

  const onMove = useCallback(
    (event: PointerEvent) => {
      const drag = dragging.current;
      if (!drag) return;
      const config = SIDES.find((item) => item.side === drag.side)!;
      const along = coordinate(config.axis === "x" ? event.clientX : event.clientY);
      const moved = ((along - drag.from) * config.sign) / drag.span;
      dispatch({ type: "set-margin", side: drag.side, fraction: drag.start + moved });
    },
    [dispatch],
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

  const frame = extendedFrame(tools.margins, picture, shown);
  return (
    <div className="studio-extend-handles">
      <svg className="studio-extend-frame" aria-hidden="true">
        <path className="studio-extend-added" fillRule="evenodd" d={`${outline(frame)} ${outline(shown)}`} />
        <rect
          className="studio-extend-edge"
          x={frame.x}
          y={frame.y}
          width={frame.width}
          height={frame.height}
        />
      </svg>
      {SIDES.map(({ side, label, axis }) => (
        <button
          key={side}
          type="button"
          className={`studio-extend-handle ${side} ${live === side ? "dragging" : ""}`}
          data-side={side}
          aria-label={`${label}, currently ${Math.round(tools.margins[side] * 100)} percent`}
          style={gripStyle(side, frame)}
          onPointerDown={(event) => {
            // The canvas underneath pans on a press; this one is the edge's.
            event.stopPropagation();
            // Measured against the picture as shown when the drag starts, so
            // a zoom partway through does not rescale what was already dragged.
            const span = axis === "x" ? shown.width : shown.height;
            if (!(span > 0)) return;
            event.preventDefault();
            (event.target as Element).setPointerCapture?.(event.pointerId);
            dragging.current = {
              side,
              from: coordinate(axis === "x" ? event.clientX : event.clientY),
              start: tools.margins[side],
              span,
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
            // The canvas pans on the same keys.
            event.stopPropagation();
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
