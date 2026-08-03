import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { toAlphaImageData, type MaskRaster } from "./studioMasks";
import type { PointerTool } from "./studioTools";
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
  onStrokeEnd,
}: {
  image: ImageBitmap | null;
  mask: MaskRaster | null;
  /** The active pointer tool; null makes the canvas view-only. */
  tool: PointerTool | null;
  /** Bump to trigger a mask repaint after undo/redo or programmatic edits. */
  maskVersion?: number;
  onStrokeEnd?: () => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageLayer = useRef<HTMLCanvasElement>(null);
  const maskLayer = useRef<HTMLCanvasElement>(null);
  const interactionLayer = useRef<HTMLCanvasElement>(null);
  const [viewport, setViewport] = useState<Viewport>(identityViewport);
  const [panning, setPanning] = useState(false);
  const spaceHeld = useRef(false);
  const lastPointer = useRef<{ x: number; y: number } | null>(null);

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
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
    };
  }, []);

  const screenPoint = (event: ReactPointerEvent | ReactWheelEvent) => {
    const bounds = containerRef.current?.getBoundingClientRect();
    return { x: event.clientX - (bounds?.left ?? 0), y: event.clientY - (bounds?.top ?? 0) };
  };

  const onPointerDown = (event: ReactPointerEvent) => {
    (event.target as Element).setPointerCapture?.(event.pointerId);
    const screen = screenPoint(event);
    if (spaceHeld.current || event.button === 1 || !tool) {
      setPanning(true);
      lastPointer.current = screen;
      return;
    }
    tool.down(toImagePoint(viewport, screen));
    drawPreview();
  };

  const onPointerMove = (event: ReactPointerEvent) => {
    const screen = screenPoint(event);
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
    if (panning) {
      setPanning(false);
      lastPointer.current = null;
      return;
    }
    if (!tool) return;
    const changed = tool.up(toImagePoint(viewport, screenPoint(event)));
    drawPreview();
    if (changed) onStrokeEnd?.();
  };

  const onWheel = (event: ReactWheelEvent) => {
    event.preventDefault();
    setViewport((current) =>
      zoomAbout(current, screenPoint(event), event.deltaY < 0 ? 1.2 : 1 / 1.2));
  };

  return (
    <div
      ref={containerRef}
      className="studio-canvas"
      role="application"
      aria-label="Image editing canvas"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
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
