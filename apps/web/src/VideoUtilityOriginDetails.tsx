import { useSensitiveMediaChoice } from "./sensitiveMedia";
import { seconds, type VideoUtilityOrigin } from "./videoUtilityText";

/** Where a saved frame or a trimmed video came from, and which part of it. */
export function VideoUtilityOriginDetails({ origin, sourceName }: {
  origin: VideoUtilityOrigin;
  /** The source's name, empty when it has none, or null when it is no longer stored. */
  sourceName: string | null;
}) {
  // While media is covered, names are covered too, as the library covers its own.
  const shielding = useSensitiveMediaChoice() !== "show";
  const source = sourceName === null
    ? "a video that is no longer stored"
    : shielding || !sourceName ? "a stored video" : sourceName;
  if (origin.action === "extract_frame") {
    const asked = Math.abs(origin.actual - origin.requested) >= 0.0005
      ? ` (asked for ${seconds(origin.requested)})` : "";
    return <div className="generation-details-content">
      <p>Saved as a picture from {source}.</p>
      <p>The frame at {seconds(origin.actual)}{asked}.</p>
    </div>;
  }
  const asked = `(asked for ${seconds(origin.requestedStart)} to ${seconds(origin.requestedEnd)})`;
  if (origin.exactFrames !== null) {
    const frames = `${origin.exactFrames} ${origin.exactFrames === 1 ? "frame" : "frames"}`;
    return <div className="generation-details-content">
      <p>Trimmed from {source}, re-encoded to start and end on the chosen frames.</p>
      <p>The {frames} from {seconds(origin.actualStart)} to {seconds(origin.actualEnd)} {asked}.</p>
    </div>;
  }
  const from = origin.fromBeginning ? "the beginning" : seconds(origin.actualStart);
  return <div className="generation-details-content">
    <p>Trimmed from {source}, copied without re-encoding.</p>
    <p>The part from {from} to {seconds(origin.actualEnd)} {asked}.</p>
  </div>;
}
