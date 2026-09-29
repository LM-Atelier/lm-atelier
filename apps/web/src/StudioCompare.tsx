import { Columns2, Eye } from "lucide-react";
import type { KeyboardEvent, PointerEvent } from "react";

/** Comparing a result with the picture it was made from, on the canvas itself.
 *
 * Holding shows the earlier picture in the result's place for as long as the
 * button is held, which is how a change too small to point at gets seen: the
 * eye catches what moves. Split lays the two across a divider instead, for a
 * change worth studying rather than spotting. Both keep the canvas's zoom, so
 * a detail is compared at the size it was edited at.
 */
export function StudioCompare({
  holding,
  onHold,
  split,
  onSplit,
  canSplit,
}: {
  holding: boolean;
  onHold: (held: boolean) => void;
  /** Where the divider stands, as a fraction of the width; null when not split. */
  split: number | null;
  onSplit: (split: number | null) => void;
  /** Only two pictures of one shape line up across a divider. */
  canSplit: boolean;
}) {
  const release = () => onHold(false);
  const holds = (key: string) => key === " " || key === "Enter";
  return (
    <div className="studio-compare" role="group" aria-label="Compare with the earlier picture">
      <button
        type="button"
        className="secondary compact-button"
        aria-pressed={holding}
        title="Shows the picture this was made from while held"
        onPointerDown={(event: PointerEvent<HTMLButtonElement>) => {
          if (event.button !== 0) return;
          // Captured, so the release is heard even off the button.
          event.currentTarget.setPointerCapture?.(event.pointerId);
          onHold(true);
        }}
        onPointerUp={release}
        onPointerCancel={release}
        onLostPointerCapture={release}
        onKeyDown={(event: KeyboardEvent<HTMLButtonElement>) => {
          if (!holds(event.key)) return;
          event.preventDefault();
          // A held key repeats, and the picture is already showing.
          if (!event.repeat) onHold(true);
        }}
        onKeyUp={(event: KeyboardEvent<HTMLButtonElement>) => {
          if (!holds(event.key)) return;
          event.preventDefault();
          release();
        }}
        onBlur={release}
      >
        <Eye size={14} aria-hidden="true" /> Hold to compare
      </button>
      {canSplit && (
        <button
          type="button"
          className="secondary compact-button"
          aria-pressed={split !== null}
          onClick={() => onSplit(split === null ? 0.5 : null)}
        >
          <Columns2 size={14} aria-hidden="true" /> Split
        </button>
      )}
      {canSplit && split !== null && (
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(split * 100)}
          aria-label="Divider position"
          onChange={(event) => onSplit(Number(event.target.value) / 100)}
        />
      )}
    </div>
  );
}
