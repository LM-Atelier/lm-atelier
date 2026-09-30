import { Columns2, Diff, Eye } from "lucide-react";
import type { KeyboardEvent, PointerEvent } from "react";
import type { StudioCompareChoice } from "./useStudioCompare";

/** Comparing a result with the picture it was made from, on the canvas itself.
 *
 * Holding shows the earlier picture in the result's place for as long as the
 * button is held, which is how a change too small to point at gets seen: the
 * eye catches what moves. Split lays the two across a divider instead, for a
 * change worth studying rather than spotting. What changed tints every pixel
 * the edit changed, and says how much of the picture that is, for an edit
 * meant to leave most of it alone. All keep the canvas's zoom, so a detail is
 * compared at the size it was edited at. Another picture from the strip can
 * stand in for the earlier one, so two results of one edit can be set against
 * each other the same ways.
 */
export function StudioCompare({
  holding,
  onHold,
  split,
  onSplit,
  canSplit,
  difference = false,
  onDifference,
  canDiffer = false,
  changed = null,
  against = null,
  choices = [],
  onAgainst,
}: {
  holding: boolean;
  onHold: (held: boolean) => void;
  /** Where the divider stands, as a fraction of the width; null when not split. */
  split: number | null;
  onSplit: (split: number | null) => void;
  /** Only two pictures of one shape line up across a divider. */
  canSplit: boolean;
  /** Whether the changed pixels are tinted over the result. */
  difference?: boolean;
  onDifference?: (on: boolean) => void;
  /** Only two pictures of exactly one size compare pixel for pixel. */
  canDiffer?: boolean;
  /** How much of the picture changed, once that is known. */
  changed?: string | null;
  /** The picture chosen to compare with, when it is not the one this was made from. */
  against?: string | null;
  choices?: StudioCompareChoice[];
  onAgainst?: (artifactId: string | null) => void;
}) {
  const release = () => onHold(false);
  const holds = (key: string) => key === " " || key === "Enter";
  return (
    <div className="studio-compare" role="group" aria-label="Compare">
      {choices.length > 0 && onAgainst && (
        <select aria-label="Compare with" value={against ?? ""} onChange={(event) => onAgainst(event.target.value || null)}>
          <option value="">What it was made from</option>
          {choices.map((choice) => (
            <option key={choice.artifactId} value={choice.artifactId}>{choice.label}</option>
          ))}
        </select>
      )}
      <button
        type="button"
        className="secondary compact-button"
        aria-pressed={holding}
        title={against ? "Shows the chosen picture while held" : "Shows the picture this was made from while held"}
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
      {canDiffer && onDifference && (
        <button
          type="button"
          className="secondary compact-button"
          aria-pressed={difference}
          title={against ? "Tints every pixel that differs from the chosen picture" : "Tints every pixel the edit changed"}
          onClick={() => onDifference(!difference)}
        >
          <Diff size={14} aria-hidden="true" /> What changed
        </button>
      )}
      {canDiffer && difference && <small aria-live="polite">{changed ?? "Comparing…"}</small>}
    </div>
  );
}
