import { useState } from "react";
import { StudioAnchorPicker } from "./StudioAnchorPicker";
import {
  EXTEND_SHAPES,
  extendedSize,
  marginsForShape,
  marginsForSize,
  type ExtendMargins,
  type ExtendSide,
} from "./studioExtend";
import type { StudioCanvasAnchor } from "./types";

const SIDES: readonly ExtendSide[] = ["top", "right", "bottom", "left"];

function whole(text: string): number | null {
  const value = Number(text);
  return text.trim() !== "" && Number.isInteger(value) && value >= 1 ? value : null;
}

function same(one: ExtendMargins, other: ExtendMargins): boolean {
  return SIDES.every((side) => Math.abs(one[side] - other[side]) < 1e-9);
}

/** Extend: how far each edge goes, by a drag on the canvas, a shape or a size.
 *
 * A shape adds canvas evenly on the two edges that grow, and a size puts the
 * new room away from where the picture is anchored. Neither ever cuts the
 * picture, and no edge gains more than twice it, which the server refuses.
 */
export function StudioExtendTool({
  margins,
  size,
  onMargins,
  onReset,
}: {
  margins: ExtendMargins;
  /** The picture's size in its own pixels, once it has been read. */
  size: { width: number; height: number } | null;
  onMargins: (margins: ExtendMargins) => void;
  onReset: () => void;
}) {
  const [anchor, setAnchor] = useState<StudioCanvasAnchor>("center");
  const extending = SIDES.some((side) => margins[side] > 0);
  const extended = size && extending ? extendedSize(margins, size) : null;
  const target = extended ?? size;
  return (
    <div className="studio-tool-options">
      <span>
        <strong>Extend by</strong>
      </span>
      <small>
        {extending
          ? SIDES.filter((side) => margins[side] > 0)
              .map((side) => `${side} ${Math.round(margins[side] * 100)}%`)
              .join(", ")
          : "Drag an edge of the picture outward, or use the arrow keys on one."}
      </small>
      {size && extended && (
        <small>{`The canvas goes from ${size.width} × ${size.height} to ${extended.width} × ${extended.height}.`}</small>
      )}
      {size && target && (
        <>
          <div className="segmented" role="group" aria-label="Extend to a shape">
            {EXTEND_SHAPES.map((shape) => {
              const reached = marginsForShape(size, shape);
              const already = reached !== null && !SIDES.some((side) => reached[side] > 0);
              const chosen = reached !== null && !already && same(reached, margins);
              return (
                <button
                  key={shape.label}
                  type="button"
                  aria-pressed={chosen}
                  className={chosen ? "active" : ""}
                  // Not disabled: a focused button that becomes disabled drops
                  // focus to the page, and a keyboard user would lose their place.
                  aria-disabled={reached === null || already}
                  title={
                    reached === null
                      ? "Farther than one extension reaches: no edge gains more than twice the picture"
                      : already
                        ? "The picture is this shape already"
                        : undefined
                  }
                  onClick={() => {
                    if (reached && !already) onMargins(reached);
                  }}
                >
                  {shape.label}
                </button>
              );
            })}
          </div>
          <ExtendToSize
            // Mounted afresh whenever the edges change, so it starts from the size they make.
            key={`${target.width}x${target.height}`}
            picture={size}
            from={target}
            anchor={anchor}
            onAnchor={setAnchor}
            onMargins={onMargins}
          />
        </>
      )}
      <button className="secondary compact-button" onClick={onReset}>
        Reset edges
      </button>
    </div>
  );
}

/** An exact size to extend to, and where the picture sits on it. */
function ExtendToSize({
  picture,
  from,
  anchor,
  onAnchor,
  onMargins,
}: {
  picture: { width: number; height: number };
  from: { width: number; height: number };
  anchor: StudioCanvasAnchor;
  onAnchor: (anchor: StudioCanvasAnchor) => void;
  onMargins: (margins: ExtendMargins) => void;
}) {
  const [width, setWidth] = useState(String(from.width));
  const [height, setHeight] = useState(String(from.height));
  const wanted = { width: whole(width), height: whole(height) };
  const size = wanted.width !== null && wanted.height !== null ? { width: wanted.width, height: wanted.height } : null;
  const smaller = size !== null && (size.width < picture.width || size.height < picture.height);
  const margins = size ? marginsForSize(picture, size, anchor) : null;
  const ready = margins !== null && SIDES.some((side) => margins[side] > 0);
  return (
    <>
      <div className="studio-resize-size">
        <label>
          <span>Width</span>
          <input type="number" min={picture.width} step={1} inputMode="numeric" value={width}
            onChange={(event) => setWidth(event.target.value)} />
        </label>
        <label>
          <span>Height</span>
          <input type="number" min={picture.height} step={1} inputMode="numeric" value={height}
            onChange={(event) => setHeight(event.target.value)} />
        </label>
      </div>
      <StudioAnchorPicker value={anchor} label="Where the picture sits" onChange={onAnchor} />
      {size === null ? (
        <small>Enter a width and a height in whole pixels.</small>
      ) : smaller ? (
        <small>Extend only adds canvas. To make the picture smaller, crop it.</small>
      ) : margins === null ? (
        <small>That is farther than one extension reaches: no edge gains more than twice the picture.</small>
      ) : null}
      <button
        type="button"
        className="secondary compact-button"
        aria-disabled={!ready}
        onClick={() => {
          if (ready && margins) onMargins(margins);
        }}
      >
        Extend to this size
      </button>
    </>
  );
}
