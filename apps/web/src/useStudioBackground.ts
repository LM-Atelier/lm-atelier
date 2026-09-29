import { useEffect, useRef, useState } from "react";
import type { StudioApplyPlan } from "./studioApplyPlan";
import { cutoutOutcome, readCutoutMask, subjectReach } from "./studioBackground";
import { dilate, encodeMaskPng, MAX_FEATHER_PX } from "./studioMasks";
import type { ChatDetail, TurnAccepted } from "./types";
import type { StudioMaskUpload } from "./useStudioSession";

type Apply = (
  instruction: string,
  artifactId: string,
  mask?: StudioMaskUpload,
  settings?: Record<string, unknown>,
  workflowRevisionId?: string,
  onAccepted?: (accepted: TurnAccepted) => void,
  secondPicture?: Blob,
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

/** Replace a background or a subject in two applies: cut the subject out, then redraw.
 *
 * The cutout is an ordinary turn, so it lands in the filmstrip like any other
 * result. Its alpha becomes the selection for the second turn, which redraws
 * the picture from the person's words and is placed back through that
 * selection. Inverted, the selection is everything around the subject, which
 * keeps its own pixels. As it is, grown a little, it is the subject, which is
 * redrawn from a second picture while everything around it stays.
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
    const started = active;
    const reference = started.plan.cutout?.reference;
    const subject = started.plan.cutout?.redraw === "subject";
    const reach = subjectReach(started.width, started.height);
    const settle = () => setReplacement((current) => (current === started ? null : current));
    void readCutoutMask(outcome.artifactId, started.width, started.height)
      .then((mask) => {
        if (!mask) return null;
        if (subject) dilate(mask, reach);
        return encodeMaskPng(mask);
      })
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
            subject
              ? // The subject, grown so a new one has room, and softened by half
                // that reach so the redrawn part meets the rest gradually. The
                // second picture is only read, never placed back into.
                {
                  blob,
                  featherPx: Math.min(MAX_FEATHER_PX, Math.round(reach / 2)),
                  invert: false,
                  apply: "blend",
                  references: reference ? 1 : 0,
                }
              : // The subject's coverage, inverted: everything around it is
                // redrawn. Its alpha is already soft, so no further feathering.
                { blob, featherPx: 0, invert: true, apply: "blend" },
            started.plan.settings,
            started.plan.workflowRevisionId,
            () => {
              settle();
              started.onAccepted();
            },
            reference,
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

  // A failed cutout has already said so and no longer holds the studio.
  const busy = active !== null && outcome?.state !== "failed";
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
      // A subject with nothing to replace it from would redraw from the words alone.
      if (cutout.redraw === "subject" && !cutout.reference) return;
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
