import { useEffect, useState } from "react";
import { sameShape } from "./studioComparison";
import { changedWords, differenceOverlay } from "./studioDifference";
import { readSourcePixels } from "./studioSourcePixels";
import { useStudioImage } from "./useStudioImage";
import type { StudioStep } from "./useStudioSession";

/** Another picture in the strip that a result can be compared with. */
export type StudioCompareChoice = { artifactId: string; label: string };

/** Comparing the result on the canvas with the picture it was made from, or with another in the strip.
 *
 * The earlier picture is decoded before anyone asks, so holding shows it at
 * once, and only one is ever kept: moving to another result releases it. A
 * picture chosen from the strip takes the earlier picture's place for that
 * result only, so results made from one picture can be set against each other.
 */
export function useStudioCompare(
  current: StudioStep | null,
  shown: ImageBitmap | null,
  previewing: boolean,
  steps: readonly StudioStep[] = [],
) {
  const currentId = current?.artifactId ?? null;
  // Chosen for one result, so moving to another compares it with what it was made from again.
  const [chosen, setChosen] = useState<{ result: string; against: string } | null>(null);
  const against = chosen !== null && chosen.result === currentId ? chosen.against : null;
  const beforeId = !previewing && current && !current.isSource ? against ?? current.beforeArtifactId : null;
  const earlier = useStudioImage(beforeId);
  const before = beforeId ? earlier.bitmap : null;
  // Held for one result, so moving to another never arrives mid-comparison.
  const [heldFor, setHeldFor] = useState<string | null>(null);
  const [split, setSplit] = useState<number | null>(null);
  // Asked for one result too, so another result never shows this one's changes.
  const [differenceFor, setDifferenceFor] = useState<string | null>(null);
  const holding = heldFor !== null && heldFor === currentId;
  const canSplit = Boolean(shown && before && sameShape(before, shown));
  // Pixel for pixel only at exactly one size; see studioDifference.
  const canDiffer = Boolean(shown && before && shown.width === before.width && shown.height === before.height);
  const differing = canDiffer && differenceFor !== null && differenceFor === currentId;
  const overlay = useDifferenceOverlay(differing ? before : null, differing ? shown : null);
  const reveal = holding ? 1 : canSplit && split !== null ? split : 0;
  return {
    /** Null until both pictures are ready, so nothing offers a comparison it
     * cannot show: a result that did not decode has no canvas to lay it over. */
    layer:
      before && shown
        ? !holding && overlay
          ? { image: overlay.bitmap, reveal: 1 }
          : { image: before, reveal }
        : null,
    controls: {
      holding,
      onHold: (held: boolean) => setHeldFor(held ? currentId : null),
      split,
      onSplit: (next: number | null) => {
        setSplit(next);
        // One comparison across the canvas at a time.
        if (next !== null) setDifferenceFor(null);
      },
      canSplit,
      difference: differing,
      onDifference: (on: boolean) => {
        setDifferenceFor(on ? currentId : null);
        if (on) setSplit(null);
      },
      canDiffer,
      changed: overlay?.words ?? null,
      against,
      choices: current && !current.isSource ? compareChoices(steps, current) : [],
      onAgainst: (artifactId: string | null) =>
        setChosen(artifactId && currentId ? { result: currentId, against: artifactId } : null),
    },
  };
}

/** The other pictures in the strip, each once: not the result itself, nor the one it was made from, which is already the default. */
function compareChoices(steps: readonly StudioStep[], current: StudioStep): StudioCompareChoice[] {
  const seen = new Set([current.artifactId, current.beforeArtifactId]);
  const choices: StudioCompareChoice[] = [];
  steps.forEach((step, index) => {
    if (seen.has(step.artifactId)) return;
    seen.add(step.artifactId);
    const words = step.instruction ? ` · ${step.instruction}` : "";
    choices.push({ artifactId: step.artifactId, label: step.isSource ? "The original" : `Step ${index}${words}` });
  });
  return choices;
}

/** The overlay of what changed between two pictures, made once and released when it goes.
 *
 * Made after the render that asks for it, since reading every pixel of two
 * pictures takes a moment. The bitmap is closed when the pair changes, when
 * the overlay is no longer asked for, and when the studio closes, and one
 * finished after it was no longer wanted is closed at once.
 */
function useDifferenceOverlay(before: ImageBitmap | null, after: ImageBitmap | null) {
  const [made, setMade] = useState<{
    before: ImageBitmap;
    after: ImageBitmap;
    bitmap: ImageBitmap;
    words: string;
  } | null>(null);

  useEffect(() => {
    if (!before || !after) return;
    let live = true;
    let owned: ImageBitmap | null = null;
    void Promise.resolve()
      .then(async () => {
        const earlier = readSourcePixels(before);
        const later = readSourcePixels(after);
        if (!live || !earlier || !later || earlier.length !== later.length) return;
        const { pixels, changed } = differenceOverlay(earlier, later);
        const bitmap = await createImageBitmap(new ImageData(pixels, after.width, after.height));
        if (!live) {
          bitmap.close();
          return;
        }
        owned = bitmap;
        setMade({ before, after, bitmap, words: changedWords(changed, after.width * after.height) });
      })
      // Nothing to show is the whole of the failure: the result stays as it is.
      .catch(() => undefined);
    return () => {
      live = false;
      owned?.close();
    };
  }, [before, after]);

  return made && made.before === before && made.after === after ? made : null;
}
