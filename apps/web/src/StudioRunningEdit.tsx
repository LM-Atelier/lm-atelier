import { Square } from "lucide-react";
import { GenerationProgress } from "./GenerationProgress";
import type { StudioRunningPlace } from "./studioApplyProgress";
import type { MessagePart } from "./types";

/** A running edit's progress, and the press that stops it.
 *
 * Stopping ends the session's running work on the server, every result it
 * asked for with it, and the picture on the canvas stays as it was.
 */
export function StudioRunningEdit({
  part,
  place,
  stopping,
  onStop,
}: {
  part: MessagePart;
  /** Which result is being made, when the edit asked for more than one. */
  place?: StudioRunningPlace | null;
  stopping: boolean;
  onStop: () => void;
}) {
  return (
    <>
      <GenerationProgress part={part} place={place ? `Result ${place.index} of ${place.count}` : undefined} />
      <div className="studio-transform-actions">
        <button
          type="button"
          className="secondary compact-button"
          // Not disabled: a focused button that becomes disabled drops focus to
          // the page, and a keyboard user would lose their place.
          aria-disabled={stopping}
          onClick={() => {
            if (!stopping) onStop();
          }}
        >
          <Square size={14} aria-hidden="true" /> {stopping ? "Stopping…" : "Stop the edit"}
        </button>
      </div>
    </>
  );
}
