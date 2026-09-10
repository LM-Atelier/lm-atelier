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
  onDimensions,
}: {
  revisionId: string;
  width: unknown;
  height: unknown;
  onDimensions: (dimensions: { width: number; height: number }) => void;
}) {
  const [pending, setPending] = useState<OutputRatioPresetId | null>(null);
  const [refused, setRefused] = useState<OutputRatioPresetId | null>(null);
  const geometry = useQuery({
    queryKey: ["workflow-revision", revisionId, "output-geometry"],
    queryFn: () => api.workflowRevisionOutputGeometry(revisionId),
  });

  const capability = geometry.data;
  // Nothing to offer is not an error and gets no row. A revision with no proof
  // that width and height reach its output, and one whose bounds express no
  // ratio exactly, both land here - and in both cases the number boxes below
  // are still the honest way to ask for a size.
  if (!capability?.available || capability.preset_ids.length === 0) return null;

  const offered = capability.preset_ids;
  const selected = ratioOf(width, height, offered);
  const resolved = Number.isInteger(width) && Number.isInteger(height)
    ? `${width as number} × ${height as number}`
    : null;

  const choose = async (preset: OutputRatioPresetId) => {
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
      setRefused(preset);
    } finally {
      setPending(null);
    }
  };

  return (
    <div className="setting-row output-ratio-control">
      <span>
        <strong>Shape</strong>
        {resolved && <small>{`Output: ${resolved}`}</small>}
        {refused && <small>{`This workflow no longer offers ${refused}.`}</small>}
      </span>
      <div className="segmented compact" role="group" aria-label="Output aspect ratio">
        {offered.map((preset) => (
          <button
            key={preset}
            type="button"
            aria-pressed={selected === preset}
            className={selected === preset ? "active" : ""}
            disabled={pending !== null}
            onClick={() => void choose(preset)}
          >
            {`${preset} ${RATIO_LABELS[preset]}`}
          </button>
        ))}
      </div>
    </div>
  );
}
