import { useEffect, useRef, useState } from "react";
import type { StudioApplyPlan } from "./studioApplyPlan";
import { cutoutOutcome, readCutoutMask } from "./studioBackground";
import { encodeMaskPng } from "./studioMasks";
import type { ChatDetail, TurnAccepted } from "./types";
import type { StudioMaskUpload } from "./useStudioSession";

type Apply = (
  instruction: string,
  artifactId: string,
  mask?: StudioMaskUpload,
  settings?: Record<string, unknown>,
  workflowRevisionId?: string,
  onAccepted?: (accepted: TurnAccepted) => void,
  secondPicture?: Blob | string,
  onRefused?: () => void,
) => void;

type Replacement = {
  /** The session it started in: another picture's session never shows its cutout. */
  sessionId: string;
  plan: StudioApplyPlan;
  sourceArtifactId: string;
  width: number;
  height: number;
  onAccepted: () => void;
  /** The cutout turn's reply, once the turn is taken. */
  cutoutMessageId: string | null;
};

export const CUTOUT_FAILED = "The subject could not be cut out, so the picture was left as it was.";
export const CUTOUT_UNREADABLE = "The cutout could not be read, so the picture was left as it was.";

/** Replace a background in two applies: cut the subject out, then redraw around it.
 *
 * The cutout is an ordinary turn, so it lands in the filmstrip like any other
 * result. Its alpha, inverted, becomes the selection for the second turn,
 * which redraws the picture from the person's words and is placed back
 * through that selection: everything around the subject changes, and the
 * subject keeps its own pixels.
 */
export function useStudioBackground(
  sessionId: string | null,
  session: ChatDetail | null,
  apply: Apply,
  onError: (message: string) => void,
) {
  const [replacement, setReplacement] = useState<Replacement | null>(null);
  // Derived, never reset: a replacement from another picture's session is not
  // this one's, and waiting on it would hold the studio busy for good.
  const active = replacement && replacement.sessionId === sessionId ? replacement : null;
  const outcome = active?.cutoutMessageId ? cutoutOutcome(session, active.cutoutMessageId) : null;
  // The cutout acted on, so a later render never redraws or reports twice.
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
      onError(CUTOUT_FAILED);
      return;
    }
    // Stopped by the person, so there is nothing to say and nothing to redraw.
    if (outcome.state === "stopped") return;
    const started = active;
    const settle = () => setReplacement((current) => (current === started ? null : current));
    void readCutoutMask(outcome.artifactId, started.width, started.height)
      .then((mask) => (mask ? encodeMaskPng(mask) : null))
      .then(
        (blob) => {
          if (!mounted.current) return;
          if (!blob) {
            settle();
            onError(CUTOUT_UNREADABLE);
            return;
          }
          apply(
            started.plan.words,
            started.sourceArtifactId,
            // The subject's coverage, inverted: everything around it is
            // redrawn. Its alpha is already soft, so no further feathering.
            { blob, featherPx: 0, invert: true, apply: "blend" },
            started.plan.settings,
            started.plan.workflowRevisionId,
            () => {
              settle();
              started.onAccepted();
            },
            undefined,
            settle,
          );
        },
        () => {
          if (!mounted.current) return;
          settle();
          onError(CUTOUT_UNREADABLE);
        },
      );
  }, [active, outcome, apply, onError]);

  // A failed cutout has already said so, and a stopped one was stopped on
  // purpose; neither holds the studio any longer.
  const busy = active !== null && outcome?.state !== "failed" && outcome?.state !== "stopped";
  return {
    busy,
    start: (
      plan: StudioApplyPlan,
      sourceArtifactId: string,
      size: { width: number; height: number },
      onAccepted: () => void,
    ) => {
      const cutout = plan.cutout;
      if (!cutout || !sessionId || busy) return;
      const started: Replacement = {
        sessionId,
        plan,
        sourceArtifactId,
        width: size.width,
        height: size.height,
        onAccepted,
        cutoutMessageId: null,
      };
      setReplacement(started);
      apply(
        cutout.words,
        sourceArtifactId,
        undefined,
        undefined,
        cutout.workflowRevisionId,
        (accepted) =>
          setReplacement((current) =>
            current === started ? { ...current, cutoutMessageId: accepted.assistant_message.id } : current),
        undefined,
        () => setReplacement((current) => (current === started ? null : current)),
      );
    },
  };
}
