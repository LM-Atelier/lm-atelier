import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { RATIO_LABELS, ratioOf } from "./outputRatio";
import type { OutputRatioPresetId } from "./types";

/** Choosing the shape of what comes out, and seeing the pixels before Send.
 *
 * Width and height already arrive as ordinary numeric fields, which is fine for
 * somebody who knows the number they want and useless for somebody who wants a
 * widescreen picture. The ratio is the thing people actually choose; the pixels
 * are a consequence of it and of what the workflow can do.
 *
 * Every number here comes from the server. The workflow proves which ratios its
 * own bounds and multiples can express exactly, and resolving one returns the
 * pair it means - so this asks rather than divides. A ratio the workflow cannot
 * hit exactly is not offered at all, because returning a picture of a shape
 * nobody asked for is worse than not offering the choice.
 */

export function OutputRatioControl({
  revisionId,
  width,
  height,
  sizeIsTheWorkflowsOwn,
  onDimensions,
}: {
  revisionId: string;
  width: unknown;
  height: unknown;
  /** The panel is offering no width and no height, so nothing here can be set. */
  sizeIsTheWorkflowsOwn: boolean;
  onDimensions: (dimensions: { width: number; height: number }) => void;
}) {
  const [pending, setPending] = useState<OutputRatioPresetId | null>(null);
  // A refusal is a fact about one revision, so it is remembered with the
  // revision it came from. The panel's role tabs swap which revision this row
  // describes without remounting it, and a message carried across that swap
  // would say a shape is gone while the button offering it sits alongside.
  const [refused, setRefused] = useState<
    { revisionId: string; preset: OutputRatioPresetId } | null
  >(null);
  const geometry = useQuery({
    queryKey: ["workflow-revision", revisionId, "output-geometry"],
    queryFn: () => api.workflowRevisionOutputGeometry(revisionId),
  });

  const capability = geometry.data;
  // Nothing to offer is not an error. A revision with no proof that width and
  // height reach its output, and one whose bounds express no ratio exactly,
  // both land here - and where the number boxes below are still offered, they
  // remain the honest way to ask for a size, so this row simply stays away.
  //
  // Where they are NOT offered there is nothing left, and silence becomes its
  // own answer: a person sees no shapes, no dimensions and no reason, and is
  // left to guess whether the app is broken. Saying that the workflow decides
  // is a statement about what this panel is showing, which is the only thing
  // that can be known here - the server answers "unsupported" without saying
  // why, on purpose, so that a caller cannot learn the shape of a graph it
  // cannot see.
  if (!capability?.available || capability.preset_ids.length === 0) {
    if (!sizeIsTheWorkflowsOwn) return null;
    return (
      <div className="setting-row output-ratio-control">
        <span>
          <strong>Shape</strong>
          <small>This workflow sets the picture size itself.</small>
        </span>
      </div>
    );
  }

  const offered = capability.preset_ids;
  const selected = ratioOf(width, height, offered);
  const resolved = Number.isInteger(width) && Number.isInteger(height)
    ? `${width as number} × ${height as number}`
    : null;
  const refusal = refused?.revisionId === revisionId ? refused.preset : null;

  const choose = async (preset: OutputRatioPresetId) => {
    // The guard lives here rather than on a disabled attribute. Disabling a
    // button that currently holds focus makes the browser drop focus to the
    // document body and never give it back, which costs a keyboard user their
    // place in the page - and with it the announcement of the choice they just
    // made. Refusing the second press is the same protection without that.
    if (pending !== null) return;
    setPending(preset);
    setRefused(null);
    try {
      const answer = await api.resolveWorkflowRevisionOutputGeometry(revisionId, {
        mode: "image",
        size_mode: "preset",
        preset_id: preset,
      });
      onDimensions({ width: answer.width, height: answer.height });
    } catch {
      // The offer came from the same proof that resolves it, so a refusal here
      // means the revision changed underneath this panel. Saying so beats
      // leaving a button that appears to do nothing.
      setRefused({ revisionId, preset });
    } finally {
      setPending(null);
    }
  };

  return (
    <div className="setting-row output-ratio-control">
      <span>
        <strong>Shape</strong>
        {resolved && <small>{`Output: ${resolved}`}</small>}
        {refusal && (
          <small role="alert">{`This workflow no longer offers ${refusal}.`}</small>
        )}
      </span>
      <div className="segmented compact" role="group" aria-label="Output aspect ratio">
        {offered.map((preset) => (
          <button
            key={preset}
            type="button"
            aria-pressed={selected === preset}
            className={selected === preset ? "active" : ""}
            aria-disabled={pending !== null}
            onClick={() => void choose(preset)}
          >
            {`${preset} ${RATIO_LABELS[preset]}`}
          </button>
        ))}
      </div>
    </div>
  );
}
