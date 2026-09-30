import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  type WheelEvent as ReactWheelEvent,
} from "react";
import {
  halfViewport,
  panSharedBy,
  WHOLE_VIEW,
  zoomSharedAbout,
  type SharedView,
  type Size,
} from "./studioSharedView";

/** One of the two pictures, and what to call it. */
export type StudioSide = { image: ImageBitmap; label: string };

const NO_SIZE: Size = { width: 0, height: 0 };

/** Two pictures next to each other, zoomed and moved as one.
 *
 * Split and holding lay one picture over the other, which needs two of one
 * shape. Side by side does not: each picture fits its own half, and a zoom or
 * a drag on either half moves both, so the same part of the scene stays in
 * view in each. Two of one shape show the same pixels at the same size. Arrow
 * keys move both, plus and minus zoom, and zero shows each whole again.
 */
export function StudioSideBySide({ before, after }: { before: StudioSide; after: StudioSide }) {
  const [view, setView] = useState<SharedView>(WHOLE_VIEW);
  const [half, setHalf] = useState<Size>(NO_SIZE);
  const firstHalf = useRef<HTMLDivElement>(null);
  const dragging = useRef<{ pointerId: number; x: number; y: number } | null>(null);

  // Both halves are one size: the grid gives them equal columns.
  useEffect(() => {
    const element = firstHalf.current;
    if (!element) return;
    const measure = () => setHalf({ width: element.clientWidth, height: element.clientHeight });
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const middle = { x: half.width / 2, y: half.height / 2 };
  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 80 : 20;
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [step, 0],
      ArrowRight: [-step, 0],
      ArrowUp: [0, step],
      ArrowDown: [0, -step],
    };
    const move = moves[event.key];
    if (move) {
      event.preventDefault();
      setView((current) => panSharedBy(current, after.image, half, move[0], move[1]));
    } else if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      setView((current) => zoomSharedAbout(current, after.image, half, middle, 1.2));
    } else if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      setView((current) => zoomSharedAbout(current, after.image, half, middle, 1 / 1.2));
    } else if (event.key === "0") {
      event.preventDefault();
      setView(WHOLE_VIEW);
    }
  };

  return (
    /* eslint-disable-next-line jsx-a11y-x/no-noninteractive-element-interactions */
    <div
      className="studio-side-by-side"
      /* A focusable viewer that takes its own keys, as the canvas is. */
      /* eslint-disable-next-line jsx-a11y-x/no-noninteractive-tabindex */
      tabIndex={0}
      role="application"
      aria-roledescription="Side by side comparison"
      aria-label={`${before.label} beside ${after.label}. Arrow keys move both, plus and minus zoom, zero shows each whole.`}
      onKeyDown={onKeyDown}
    >
      {[before, after].map((side, index) => (
        <StudioSideHalf
          key={index}
          side={side}
          view={view}
          half={half}
          measured={index === 0 ? firstHalf : undefined}
          onWheel={(event, anchor) => {
            event.preventDefault();
            setView((current) => zoomSharedAbout(current, side.image, half, anchor, event.deltaY < 0 ? 1.2 : 1 / 1.2));
          }}
          onPointerDown={(event) => {
            if (event.button !== 0) return;
            event.currentTarget.setPointerCapture?.(event.pointerId);
            dragging.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
          }}
          onPointerMove={(event) => {
            const drag = dragging.current;
            if (!drag || drag.pointerId !== event.pointerId) return;
            const dx = event.clientX - drag.x;
            const dy = event.clientY - drag.y;
            dragging.current = { ...drag, x: event.clientX, y: event.clientY };
            setView((current) => panSharedBy(current, side.image, half, dx, dy));
          }}
          onPointerEnd={(event) => {
            if (dragging.current?.pointerId === event.pointerId) dragging.current = null;
          }}
        />
      ))}
    </div>
  );
}

/** One half: its picture under the shared view, and its name. */
function StudioSideHalf({
  side,
  view,
  half,
  measured,
  onWheel,
  onPointerDown,
  onPointerMove,
  onPointerEnd,
}: {
  side: StudioSide;
  view: SharedView;
  half: Size;
  measured?: RefObject<HTMLDivElement | null>;
  onWheel: (event: ReactWheelEvent, anchor: { x: number; y: number }) => void;
  onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onPointerMove: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onPointerEnd: (event: ReactPointerEvent<HTMLDivElement>) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, side.image.width, side.image.height);
    context.drawImage(side.image, 0, 0);
  }, [side.image]);
  const viewport = halfViewport(view, side.image, half);
  return (
    <figure className="studio-side">
      <div
        ref={measured}
        className="studio-side-view"
        onWheel={(event) => {
          const bounds = event.currentTarget.getBoundingClientRect();
          const x = Number.isFinite(event.clientX) ? event.clientX - bounds.left : half.width / 2;
          const y = Number.isFinite(event.clientY) ? event.clientY - bounds.top : half.height / 2;
          onWheel(event, { x, y });
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerEnd}
        onPointerCancel={onPointerEnd}
        onLostPointerCapture={onPointerEnd}
      >
        <canvas
          ref={canvas}
          width={side.image.width}
          height={side.image.height}
          style={{
            width: side.image.width,
            height: side.image.height,
            transform: `translate(${viewport.tx}px, ${viewport.ty}px) scale(${viewport.scale})`,
          }}
        />
      </div>
      <figcaption>{side.label}</figcaption>
    </figure>
  );
}
