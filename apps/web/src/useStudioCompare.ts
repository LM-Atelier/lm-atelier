import { useState } from "react";
import { sameShape } from "./studioComparison";
import { useStudioImage } from "./useStudioImage";
import type { StudioStep } from "./useStudioSession";

/** Comparing the result on the canvas with the picture it was made from.
 *
 * The earlier picture is decoded before anyone asks, so holding shows it at
 * once, and only one is ever kept: moving to another result releases it.
 */
export function useStudioCompare(
  current: StudioStep | null,
  shown: ImageBitmap | null,
  previewing: boolean,
) {
  const currentId = current?.artifactId ?? null;
  const beforeId = !previewing && current && !current.isSource ? current.beforeArtifactId : null;
  const earlier = useStudioImage(beforeId);
  const before = beforeId ? earlier.bitmap : null;
  // Held for one result, so moving to another never arrives mid-comparison.
  const [heldFor, setHeldFor] = useState<string | null>(null);
  const [split, setSplit] = useState<number | null>(null);
  const holding = heldFor !== null && heldFor === currentId;
  const canSplit = Boolean(shown && before && sameShape(before, shown));
  const reveal = holding ? 1 : canSplit && split !== null ? split : 0;
  return {
    /** Null until both pictures are ready, so nothing offers a comparison it
     * cannot show: a result that did not decode has no canvas to lay it over. */
    layer: before && shown ? { image: before, reveal } : null,
    controls: {
      holding,
      onHold: (held: boolean) => setHeldFor(held ? currentId : null),
      split,
      onSplit: setSplit,
      canSplit,
    },
  };
}
