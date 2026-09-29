import { Square } from "lucide-react";
import { GenerationProgress } from "./GenerationProgress";
import type { MessagePart } from "./types";

/** A running edit's progress, and the press that stops it.
 *
 * Stopping ends the session's running work on the server, and the picture on
 * the canvas stays as it was.
 */
export function StudioRunningEdit({
  part,
  stopping,
  onStop,
}: {
  part: MessagePart;
  stopping: boolean;
  onStop: () => void;
}) {
  return (
    <>
      <GenerationProgress part={part} />
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
