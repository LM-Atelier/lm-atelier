import { Sparkles } from "lucide-react";
import type { MessagePart } from "./types";

/** A running generation's step and how far along it is, as its progress part reports them.
 *
 * Shared by the conversation and Image Studio, so an edit reads the same in
 * both places. It announces politely and never takes focus. `place` goes before
 * the step, for work that is one of several made one after another.
 */
export function GenerationProgress({ part, place }: { part: MessagePart; place?: string }) {
  const progress = Number(part.metadata_json.progress ?? 0);
  const indeterminate = part.metadata_json.indeterminate === true;
  return (
    <div className="generation-progress" role="status" aria-live="polite">
      <Sparkles size={17} />
      <div>
        <span>{place ? `${place} · ` : ""}{part.text || "Working"}</span>
        <div className="progress-track">
          <div
            className={indeterminate ? "indeterminate" : undefined}
            style={indeterminate ? undefined : { width: `${progress * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}
