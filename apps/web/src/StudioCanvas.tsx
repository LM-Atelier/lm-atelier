import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { toAlphaImageData, type MaskRaster } from "./studioMasks";
import type { ImagePoint, PointerTool } from "./studioTools";
import {
  fitViewport,
  identityViewport,
  panBy,
  toImagePoint,
  zoomAbout,
  type Viewport,
} from "./studioViewport";

/** The studio's three-layer canvas: image, mask tint, live interaction.
 *
 * All geometry lives in the pure viewport module and all mask mutation in
 * the pure tools; this component is deliberately thin glue - it unprojects
 * pointer events, forwards them to the active tool, and repaints layers.
 * Space-drag pans, wheel zooms about the cursor, and the container's CSS
 * transform carries the one shared viewport so layers can never disagree.
 */
export function StudioCanvas({
  image,
  mask,
  tool,
  maskVersion,
  onGestureStart,
  onStrokeEnd,
}: {
  image: ImageBitmap | null;
  mask: MaskRaster | null;
  /** The active pointer tool; null makes the canvas view-only. */
  tool: PointerTool | null;
  /** Bump to trigger a mask repaint after undo/redo or programmatic edits. */
  maskVersion?: number;
  /** Fires before the first mutation of a gesture: the undo snapshot point.
   * Pushing at stroke end would snapshot the already-mutated raster, making
   * the first Undo a no-op. */
  onGestureStart?: () => void;
  onStrokeEnd?: () => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageLayer = useRef<HTMLCanvasElement>(null);
  const maskLayer = useRef<HTMLCanvasElement>(null);
  const interactionLayer = useRef<HTMLCanvasElement>(null);
  const caret = useRef<ImagePoint | null>(null);
  const keyboardStroke = useRef(false);
  const [viewport, setViewport] = useState<Viewport>(identityViewport);
  const [panning, setPanning] = useState(false);
  const spaceHeld = useRef(false);
  const lastPointer = useRef<{ x: number; y: number } | null>(null);
  const activePointers = useRef<Map<number, { x: number; y: number }>>(new Map());
  const drawingPointer = useRef<number | null>(null);

  const size = useMemo(
    () => ({ width: image?.width ?? 0, height: image?.height ?? 0 }),
    [image],
  );

  // Fit on image arrival; the container is the sizing authority.
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !image) return;
    setViewport(fitViewport(image, {
      width: container.clientWidth || image.width,
      height: container.clientHeight || image.height,
    }));
  }, [image]);

  useEffect(() => {
    const layer = imageLayer.current;
    const context = layer?.getContext("2d");
    if (!layer || !context || !image) return;
    context.clearRect(0, 0, layer.width, layer.height);
    context.drawImage(image, 0, 0);
  }, [image, size]);

  useEffect(() => {
    const layer = maskLayer.current;
    const context = layer?.getContext("2d");
    if (!layer || !context) return;
    context.clearRect(0, 0, layer.width, layer.height);
    if (!mask) return;
    const tint = new ImageData(toAlphaImageData(mask, [80, 170, 255]), mask.width, mask.height);
    context.putImageData(tint, 0, 0);
  }, [mask, maskVersion, size]);

  const drawPreview = useCallback(() => {
    const layer = interactionLayer.current;
    const context = layer?.getContext("2d");
    if (!layer || !context) return;
    context.clearRect(0, 0, layer.width, layer.height);
    const preview = tool?.preview();
    if (!preview || preview.kind === "none") return;
    context.strokeStyle = "rgba(80, 170, 255, 0.9)";
    context.lineWidth = Math.max(1, 1.5 / viewport.scale);
    if (preview.kind === "brush-cursor") {
      context.beginPath();
      context.arc(preview.center.x, preview.center.y, preview.radius, 0, Math.PI * 2);
      context.stroke();
    } else if (preview.kind === "rect") {
      context.strokeRect(
        Math.min(preview.from.x, preview.to.x),
        Math.min(preview.from.y, preview.to.y),
        Math.abs(preview.to.x - preview.from.x),
        Math.abs(preview.to.y - preview.from.y),
      );
    } else {
      context.beginPath();
      preview.points.forEach((point, index) =>
        index === 0 ? context.moveTo(point.x, point.y) : context.lineTo(point.x, point.y));
      context.stroke();
    }
  }, [tool, viewport.scale]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.code === "Space") spaceHeld.current = event.type === "keydown";
    };
    // A Space release outside the window would otherwise strand pan mode.
    const onBlur = () => {
      spaceHeld.current = false;
      setPanning(false);
      lastPointer.current = null;
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
      window.removeEventListener("blur", onBlur);
    };
  }, []);

  /** Where the caret sits, put at the middle of the picture on first use. */
  const caretPoint = (): ImagePoint => {
    if (!caret.current) {
      caret.current = { x: (image?.width ?? 0) / 2, y: (image?.height ?? 0) / 2 };
    }
    return caret.current;
  };

  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 80 : 20;
    const NUDGES: Record<string, [number, number]> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, -1],
      ArrowDown: [0, 1],
    };
    const nudge = NUDGES[event.key];
    if (nudge) {
      event.preventDefault();
      // With a tool in hand the arrows move the caret, because that is the
      // thing there was no keyboard path to at all. Panning keeps them
      // under Alt, and keeps them outright when no tool is selected.
      if (!tool || event.altKey) {
        setViewport((current) => panBy(current, -nudge[0] * step, -nudge[1] * step));
        return;
      }
      const at = caretPoint();
      const moved = { x: at.x + nudge[0] * step, y: at.y + nudge[1] * step };
      caret.current = moved;
      // move() paints only while a stroke is open and merely tracks the
      // hover otherwise, so this both extends a live selection and shows
      // the caret when there is none.
      tool.move(moved);
      drawPreview();
      return;
    }
    if (tool && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      const at = caretPoint();
      if (keyboardStroke.current) {
        keyboardStroke.current = false;
        if (tool.up(at)) {
          onStrokeEnd?.();
        }
        drawPreview();
      } else {
        onGestureStart?.();
        keyboardStroke.current = true;
        tool.down(at);
        drawPreview();
      }
      return;
    }
    if (keyboardStroke.current && event.key === "Escape") {
      event.preventDefault();
      keyboardStroke.current = false;
      // Abandon rather than finish: up() on a rectangle or lasso is what
      // commits it, so cancelling through that door would paint the very
      // thing being cancelled.
      tool?.cancel();
      drawPreview();
      return;
    }
    // Zoom about the middle of the view, since there is no pointer to
    // zoom about.
    const bounds = containerRef.current?.getBoundingClientRect();
    const middle = { x: (bounds?.width ?? 0) / 2, y: (bounds?.height ?? 0) / 2 };
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      setViewport((current) => zoomAbout(current, middle, 1.2));
    } else if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      setViewport((current) => zoomAbout(current, middle, 1 / 1.2));
    } else if (event.key === "0" && image) {
      event.preventDefault();
      setViewport(fitViewport(image, {
        width: bounds?.width ?? 0,
        height: bounds?.height ?? 0,
      }));
    }
  };

  const screenPoint = (event: ReactPointerEvent | ReactWheelEvent) => {
    const bounds = containerRef.current?.getBoundingClientRect();
    // A synthetic or touch-only event can omit client coordinates; treating
    // them as zero keeps every downstream point finite.
    const clientX = Number.isFinite(event.clientX) ? event.clientX : 0;
    const clientY = Number.isFinite(event.clientY) ? event.clientY : 0;
    const left = Number.isFinite(bounds?.left) ? (bounds?.left ?? 0) : 0;
    const top = Number.isFinite(bounds?.top) ? (bounds?.top ?? 0) : 0;
    return { x: clientX - left, y: clientY - top };
  };

  const onPointerDown = (event: ReactPointerEvent) => {
    (event.target as Element).setPointerCapture?.(event.pointerId);
    const screen = screenPoint(event);
    activePointers.current.set(event.pointerId, screen);
    // A second pointer converts the gesture to pan/zoom rather than drawing
    // two strokes at once; the in-progress stroke ends where it stands. The
    // check is "a stroke is already live", not a pointer count, because a
    // second down can reuse an id and would otherwise start a second stroke.
    if (activePointers.current.size > 1 || drawingPointer.current !== null) {
      if (drawingPointer.current !== null && tool) {
        const pointerId = drawingPointer.current;
        drawingPointer.current = null;
        tool.up(toImagePoint(viewport, activePointers.current.get(pointerId) ?? screen));
        drawPreview();
      }
      setPanning(true);
      lastPointer.current = screen;
      return;
    }
    if (spaceHeld.current || event.button === 1 || !tool) {
      setPanning(true);
      lastPointer.current = screen;
      return;
    }
    drawingPointer.current = event.pointerId;
    onGestureStart?.();
    tool.down(toImagePoint(viewport, screen));
    drawPreview();
  };

  const endDrawing = (screen: { x: number; y: number }): boolean => {
    if (drawingPointer.current === null || !tool) return false;
    drawingPointer.current = null;
    const changed = tool.up(toImagePoint(viewport, screen));
    drawPreview();
    return changed;
  };

  const onPointerMove = (event: ReactPointerEvent) => {
    const screen = screenPoint(event);
    if (activePointers.current.has(event.pointerId)) {
      activePointers.current.set(event.pointerId, screen);
    }
    if (panning && lastPointer.current) {
      setViewport((current) =>
        panBy(current, screen.x - lastPointer.current!.x, screen.y - lastPointer.current!.y));
      lastPointer.current = screen;
      return;
    }
    tool?.move(toImagePoint(viewport, screen));
    drawPreview();
  };

  const onPointerUp = (event: ReactPointerEvent) => {
    const screen = screenPoint(event);
    activePointers.current.delete(event.pointerId);
    if (panning) {
      if (activePointers.current.size === 0) {
        setPanning(false);
        lastPointer.current = null;
      }
      return;
    }
    if (endDrawing(screen)) onStrokeEnd?.();
  };

  /** Cancellation and lost capture must not leave a half-drawn gesture. */
  const onPointerCancel = (event: ReactPointerEvent) => {
    activePointers.current.delete(event.pointerId);
    if (drawingPointer.current === event.pointerId) {
      endDrawing(screenPoint(event));
      onStrokeEnd?.();
    }
    if (activePointers.current.size === 0) {
      setPanning(false);
      lastPointer.current = null;
    }
  };

  const onWheel = (event: ReactWheelEvent) => {
    event.preventDefault();
    setViewport((current) =>
      zoomAbout(current, screenPoint(event), event.deltaY < 0 ? 1.2 : 1 / 1.2));
  };

  return (
    /* eslint-disable-next-line jsx-a11y-x/no-noninteractive-element-interactions */
    <div
      ref={containerRef}
      className="studio-canvas"
      /* The rules below read the implicit role of the tag, not the explicit
         one: a focusable canvas application is exactly what they exist to
         prevent being written by accident, and exactly what this is. */
      /* eslint-disable-next-line jsx-a11y-x/no-noninteractive-tabindex */
      tabIndex={0}
      // This role hands every key to the element rather than to the screen
      // reader, which was a lie while the element could not take focus and
      // handled no keys at all. It is true now: the canvas is focusable and
      // drives its own viewport. Drawing a selection still needs a pointer.
      role="application"
      aria-roledescription="Image canvas"
      aria-label={
        tool
          ? "Image editing canvas. Arrow keys move the selection point, Enter starts and finishes a selection, Escape cancels it, Alt with arrows pans, plus and minus zoom, zero fits the image."
          : "Image editing canvas. Arrow keys pan, plus and minus zoom, zero fits the image."
      }
      onKeyDown={onKeyDown}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onLostPointerCapture={onPointerCancel}
      onWheel={onWheel}
    >
      <div
        className="studio-canvas-layers"
        style={{
          transform: `translate(${viewport.tx}px, ${viewport.ty}px) scale(${viewport.scale})`,
          transformOrigin: "0 0",
          width: size.width,
          height: size.height,
        }}
      >
        <canvas ref={imageLayer} width={size.width} height={size.height} />
        <canvas ref={maskLayer} width={size.width} height={size.height} data-layer="mask" />
        <canvas
          ref={interactionLayer}
          width={size.width}
          height={size.height}
          data-layer="interaction"
        />
      </div>
    </div>
  );
}
