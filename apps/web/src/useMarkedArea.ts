import { useMemo, useState } from "react";
import { cloneMask, encodeMaskPng, feather, isEmpty, type MaskRaster } from "./studioMasks";

/** The marked area as a tool that acts on it needs it: marked or not, and as a picture.
 *
 * Blurring and painting both work on what the brush, or any other selection
 * tool, has marked. Handing it over means feathering it as the selection's
 * feather says and encoding it as a PNG, which can fail; the failure is said
 * rather than swallowed, so a press never silently does nothing.
 */
export function useMarkedArea(mask: MaskRaster | null, maskVersion: number, featherPx: number) {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const marked = useMemo(() => mask !== null && !isEmpty(mask), [mask, maskVersion]);
  const [preparing, setPreparing] = useState(false);
  const [refusal, setRefusal] = useState<string | null>(null);

  /** Encode the marked area and hand it to `use`, once it is ready. */
  const handOver = (use: (selection: Blob) => void) => {
    if (!marked || !mask || preparing) return;
    const selection = cloneMask(mask);
    if (featherPx > 0) feather(selection, featherPx);
    setPreparing(true);
    setRefusal(null);
    const refuse = () => setRefusal("The marked area could not be prepared. Mark it again.");
    void encodeMaskPng(selection).then(
      (encoded) => {
        setPreparing(false);
        if (encoded) use(encoded);
        else refuse();
      },
      () => {
        setPreparing(false);
        refuse();
      },
    );
  };

  return { marked, preparing, refusal, handOver };
}
