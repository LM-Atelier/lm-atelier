import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { CURVE_POINTS, curveLevel, toneCurve } from "./studioAdjustments";
import type { StudioCurvePoint } from "./types";

/** How near a press must land to a point, in levels, to take that point rather than add one. */
const REACH = 10;

/** The level under a pointer on the graph: across is the level read, up is what it becomes. */
function levelAt(graph: SVGSVGElement, clientX: number, clientY: number): StudioCurvePoint {
  const box = graph.getBoundingClientRect();
  const across = box.width > 0 ? (clientX - box.left) / box.width : 0;
  const up = box.height > 0 ? 1 - (clientY - box.top) / box.height : 0;
  return {
    x: Math.min(255, Math.max(0, Math.round(across * 255))),
    y: Math.min(255, Math.max(0, Math.round(up * 255))),
  };
}

/** The points with one moved to `to`, kept between its neighbours and inside the range. */
function moved(points: StudioCurvePoint[], index: number, to: StudioCurvePoint): StudioCurvePoint[] {
  const low = index > 0 ? points[index - 1].x + 1 : 1;
  const high = index < points.length - 1 ? points[index + 1].x - 1 : 254;
  const point = { x: Math.min(high, Math.max(low, to.x)), y: Math.min(255, Math.max(0, to.y)) };
  return points.map((each, at) => (at === index ? point : each));
}

/** The line the curve draws on the graph, one step to a level, black at the bottom left. */
function curvePath(points: StudioCurvePoint[]): string {
  const curve = toneCurve(points);
  const heights = Array.from({ length: 256 }, (_, level) => (curve ? curveLevel(curve, level) : level));
  return heights.map((height, level) => `${level === 0 ? "M" : "L"}${level} ${(255 - height).toFixed(2)}`).join(" ");
}

/** The picture's brightness as a shape along the bottom of the graph, tallest at the commonest level.
 *
 * Heights go by the square root of each count, so the few levels a picture
 * barely uses still show beside the ones it uses most.
 */
function histogramPath(histogram: ArrayLike<number>): string {
  let most = 0;
  for (let level = 0; level < 256; level += 1) most = Math.max(most, histogram[level] ?? 0);
  if (most === 0) return "";
  const tops = Array.from({ length: 256 }, (_, level) => 255 - Math.sqrt((histogram[level] ?? 0) / most) * 255);
  return `M0 255 ${tops.map((top, level) => `L${level} ${top.toFixed(2)}`).join(" ")} L255 255 Z`;
}

/** The tone curve: the line each level is read against, from black to white.
 *
 * Pressing on the graph adds a point there and drags it, or takes the point
 * already near the press. Double-clicking a point, or Delete while it has
 * focus, takes it away. Arrow keys move a focused point by one level, or ten
 * with Shift. A point stays between its neighbours, so the curve always runs
 * left to right, and black and white stay at the ends. The line drawn is the
 * one the picture is made with, over the picture's own brightness where it
 * can be read.
 */
export function StudioToneCurve({
  points,
  busy,
  onChange,
  histogram = null,
}: {
  points: StudioCurvePoint[];
  busy: boolean;
  onChange: (points: StudioCurvePoint[]) => void;
  /** The picture's brightness, 256 levels of counts; absent where it cannot be read. */
  histogram?: ArrayLike<number> | null;
}) {
  const graph = useRef<SVGSVGElement>(null);
  const [dragging, setDragging] = useState<number | null>(null);

  const press = (event: PointerEvent<SVGSVGElement>) => {
    const surface = graph.current;
    if (busy || event.button !== 0 || !surface) return;
    const at = levelAt(surface, event.clientX, event.clientY);
    let nearest = -1;
    points.forEach((point, index) => {
      const distance = Math.hypot(point.x - at.x, point.y - at.y);
      if (distance <= REACH && (nearest < 0 || distance < Math.hypot(points[nearest].x - at.x, points[nearest].y - at.y))) {
        nearest = index;
      }
    });
    if (nearest < 0) {
      if (points.length >= CURVE_POINTS || at.x < 1 || at.x > 254 || points.some((point) => point.x === at.x)) return;
      nearest = points.filter((point) => point.x < at.x).length;
      onChange([...points.slice(0, nearest), at, ...points.slice(nearest)]);
    }
    // Captured, so the drag goes on when the pointer leaves the graph.
    surface.setPointerCapture?.(event.pointerId);
    setDragging(nearest);
  };

  const drag = (event: PointerEvent<SVGSVGElement>) => {
    const surface = graph.current;
    if (dragging === null || !surface || dragging >= points.length) return;
    onChange(moved(points, dragging, levelAt(surface, event.clientX, event.clientY)));
  };

  const remove = (index: number) => {
    if (!busy) onChange(points.filter((_, at) => at !== index));
  };

  const nudge = (event: KeyboardEvent<SVGCircleElement>, index: number) => {
    if (busy) return;
    const step = event.shiftKey ? 10 : 1;
    const point = points[index];
    const by: Record<string, StudioCurvePoint> = {
      ArrowLeft: { x: point.x - step, y: point.y },
      ArrowRight: { x: point.x + step, y: point.y },
      ArrowUp: { x: point.x, y: point.y + step },
      ArrowDown: { x: point.x, y: point.y - step },
    };
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      remove(index);
    } else if (by[event.key]) {
      event.preventDefault();
      onChange(moved(points, index, by[event.key]));
    }
  };

  return (
    <div className="studio-tone-curve" role="group" aria-label="Tone curve">
      <span><strong>Tone curve</strong></span>
      <svg
        ref={graph}
        viewBox="0 0 255 255"
        onPointerDown={press}
        onPointerMove={drag}
        onPointerUp={() => setDragging(null)}
        onPointerCancel={() => setDragging(null)}
        onLostPointerCapture={() => setDragging(null)}
      >
        {histogram && <path className="studio-tone-curve-histogram" d={histogramPath(histogram)} />}
        <path className="studio-tone-curve-diagonal" d="M0 255 L255 0" />
        <path className="studio-tone-curve-line" d={curvePath(points)} />
        {points.map((point, index) => (
          <circle
            key={index}
            className="studio-tone-curve-point"
            cx={point.x}
            cy={255 - point.y}
            r={7}
            tabIndex={0}
            role="slider"
            aria-label={`Curve point ${index + 1}`}
            aria-valuemin={0}
            aria-valuemax={255}
            aria-valuenow={point.y}
            aria-valuetext={`${point.x} becomes ${point.y}`}
            onDoubleClick={() => remove(index)}
            onKeyDown={(event) => nudge(event, index)}
          />
        ))}
      </svg>
      <small>
        {points.length === 0
          ? "Press on the graph to add a point there, then drag it."
          : points.length < CURVE_POINTS
            ? "Drag a point, or press elsewhere on the graph to add one."
            : `A curve takes ${CURVE_POINTS} points; take one away to add another.`}
      </small>
      <button
        type="button"
        className="secondary compact-button"
        aria-disabled={points.length === 0 || busy}
        onClick={() => {
          if (points.length > 0 && !busy) onChange([]);
        }}
      >
        Straighten the curve
      </button>
    </div>
  );
}
