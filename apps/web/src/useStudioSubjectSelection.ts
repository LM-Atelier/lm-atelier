import { useEffect, useRef, useState } from "react";
import { cutoutOutcome, readCutoutMask } from "./studioBackground";
import type { MaskRaster } from "./studioMasks";
import { CUTOUT_INSTRUCTION } from "./studioToolState";
import type { ChatDetail } from "./types";
import type { useStudioSession } from "./useStudioSession";

export const SUBJECT_NOT_FOUND = "The subject could not be found, so the selection was left as it was.";
export const SUBJECT_UNREADABLE = "The subject that was found could not be read, so the selection was left as it was.";
export const SUBJECT_ELSEWHERE =
  "The subject was found in a picture that is no longer on the canvas, so the selection was left as it was.";

type Finding = {
  /** The session it started in: another picture's session never takes its subject. */
  sessionId: string;
  /** The picture the subject is looked for in, whose selection it becomes. */
  artifactId: string;
  width: number;
  height: number;
  /** The cutout turn's reply, once the turn is taken. */
  cutoutMessageId: string | null;
};

/** Select the subject: the cutout workflow finds it, and the cutout's alpha becomes the selection.
 *
 * The cutout is an ordinary turn, so it joins the strip like any other result.
 * Its alpha is read at the size of the picture the subject was looked for in,
 * soft edges and all, and handed back with that picture's name, since only the
 * selection drawn on that picture can take it. From there it is a selection
 * like any other: drawn on, inverted, softened or undone.
 */
export function useStudioSubjectSelection(
  sessionId: string | null,
  session: ChatDetail | null,
  apply: ReturnType<typeof useStudioSession>["apply"],
  onError: (message: string) => void,
  onFound: (artifactId: string, mask: MaskRaster) => void,
) {
  const [finding, setFinding] = useState<Finding | null>(null);
  // Derived, never reset: a search from another picture's session is not this
  // one's, and waiting on it would hold the studio busy for good.
  const active = finding && finding.sessionId === sessionId ? finding : null;
  const outcome = active?.cutoutMessageId ? cutoutOutcome(session, active.cutoutMessageId) : null;
  // The cutout acted on, so a later render never reads or reports it twice.
  const handled = useRef<string | null>(null);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    const cutoutMessageId = active?.cutoutMessageId;
    if (!active || !cutoutMessageId || !outcome || outcome.state === "waiting") return;
    if (handled.current === cutoutMessageId) return;
    handled.current = cutoutMessageId;
    if (outcome.state === "failed") {
      onError(SUBJECT_NOT_FOUND);
      return;
    }
    // Stopped by the person, so there is nothing to say and nothing to select.
    if (outcome.state === "stopped") return;
    const started = active;
    const settle = () => setFinding((current) => (current === started ? null : current));
    void readCutoutMask(outcome.artifactId, started.width, started.height).then(
      (mask) => {
        if (!mounted.current) return;
        settle();
        if (mask) onFound(started.artifactId, mask);
        else onError(SUBJECT_UNREADABLE);
      },
      () => {
        if (!mounted.current) return;
        settle();
        onError(SUBJECT_UNREADABLE);
      },
    );
  }, [active, outcome, onError, onFound]);

  // A cutout that failed has already said so, and one stopped was stopped on
  // purpose; neither holds the studio any longer.
  const busy = active !== null && outcome?.state !== "failed" && outcome?.state !== "stopped";
  return {
    busy,
    /** Look for the subject in a picture, on the workflow the report names for cutting one out. */
    start: (artifactId: string, size: { width: number; height: number }, workflowRevisionId: string) => {
      if (!sessionId || busy) return;
      const started: Finding = { sessionId, artifactId, width: size.width, height: size.height, cutoutMessageId: null };
      setFinding(started);
      apply(
        CUTOUT_INSTRUCTION,
        artifactId,
        undefined,
        undefined,
        workflowRevisionId,
        (accepted) =>
          setFinding((current) =>
            current === started ? { ...current, cutoutMessageId: accepted.assistant_message.id } : current),
        undefined,
        () => setFinding((current) => (current === started ? null : current)),
      );
    },
  };
}
