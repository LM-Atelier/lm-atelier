import { useMutation } from "@tanstack/react-query";
import { Dices } from "lucide-react";
import { useState } from "react";
import { api } from "./api";
import { studioRecipeSource } from "./studioRecipeSource";
import { studioReplay, type StudioReplay } from "./studioReplay";
import type { ChatDetail } from "./types";
import { studioTurnPictures, type StudioStep } from "./useStudioSession";

/** Try another: a result's own edit again, on the pictures it was given, with a new random seed.
 *
 * Offered only for a result a model made, since an exact edit would come back
 * the same. The new result joins the strip made from the same picture as this
 * one, which stays where it is: another is an alternative, not a replacement.
 * Mounted afresh for each result, so a refusal about one never shows beside
 * another.
 */
export function StudioTryAnother({
  session,
  current,
  busy,
  onTry,
}: {
  session: ChatDetail | null;
  current: StudioStep | null;
  busy: boolean;
  onTry: (replay: StudioReplay) => void;
}) {
  const source = studioRecipeSource(session, current);
  const [unreadable, setUnreadable] = useState(false);
  const read = useMutation({ mutationFn: (runId: string) => api.run(runId) });
  if (!source || !current) return null;
  const waiting = busy || read.isPending;
  return (
    <>
      <button
        type="button"
        className="secondary compact-button"
        // Not disabled: a focused button that becomes disabled drops focus to
        // the page, and a keyboard user would lose their place.
        aria-disabled={waiting}
        title="The same edit on the same picture, with a new random seed"
        onClick={() => {
          if (waiting) return;
          setUnreadable(false);
          read.mutate(source.runId, {
            onSuccess: (run) => {
              const replay = studioReplay(run, source.instruction, studioTurnPictures(session, current.messageId));
              if (replay) onTry(replay);
              else setUnreadable(true);
            },
            onError: () => setUnreadable(true),
          });
        }}
      >
        <Dices size={14} aria-hidden="true" /> Try another
      </button>
      {unreadable && <small role="status">This result's edit could not be read, so nothing was sent.</small>}
    </>
  );
}
