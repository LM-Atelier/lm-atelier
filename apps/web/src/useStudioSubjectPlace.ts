import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { StudioApplyPlan } from "./studioApplyPlan";
import { cutoutOutcome, readCutoutMask, subjectReach } from "./studioBackground";
import { dilate, encodeMaskPng, MAX_FEATHER_PX } from "./studioMasks";
import { encodeRgbaPng, placedSubject, readCutoutPixels, solidBox, type PictureBox } from "./studioPlaceSubject";
import type { ChatDetail, TurnAccepted } from "./types";
import { CUTOUT_FAILED, CUTOUT_UNREADABLE } from "./useStudioBackground";
import type { useStudioSession } from "./useStudioSession";

export const NO_SUBJECT_FOUND = "No subject was found in the picture, so it was left as it was.";
export const NEW_SUBJECT_FAILED =
  "The new subject could not be cut out of its picture, so the picture was left as it was.";
export const NO_NEW_SUBJECT = "No subject was found in the second picture, so the picture was left as it was.";
export const REMOVAL_FAILED = "The old subject could not be removed, so the picture was left as it was.";

type Session = ReturnType<typeof useStudioSession>;

/** The turn a replacement waits on, and what the turns before it found. */
type Step =
  /** Finding the old subject in the picture. */
  | { stage: "find" }
  /** Cutting the new subject out of its picture: the old one's box, and what it covered grown and encoded, are known. */
  | { stage: "cut"; box: PictureBox; removal: Blob }
  /** Removing the old subject: the new one is drawn alone where it goes. */
  | { stage: "remove"; placed: Blob };

type Replacing = {
  /** Which replacement this is: a later one never takes an earlier one's answers. */
  run: number;
  /** The session it started in: another picture's session never shows its steps. */
  sessionId: string;
  plan: StudioApplyPlan;
  sourceArtifactId: string;
  width: number;
  height: number;
  onAccepted: () => void;
  step: Step;
  /** The reply this step waits on, once its turn is taken. */
  messageId: string | null;
};

/** Replace a subject: find the old one, cut the new one out, remove the old one, and place the new one where it stood.
 *
 * Every model step is an ordinary turn, so each lands in the strip like any
 * other result. The new subject is cut out before anything is removed, so a
 * second picture with nothing in it changes nothing. The last step needs no
 * model: the browser scales the new subject into the old one's box and the
 * server lays it over the picture the old one was removed from.
 */
export function useStudioSubjectPlace(
  sessionId: string | null,
  session: ChatDetail | null,
  apply: Session["apply"],
  localEdit: Session["localEdit"],
  onError: (message: string) => void,
) {
  const [replacing, setReplacing] = useState<Replacing | null>(null);
  // Derived, never reset: a replacement from another picture's session is not
  // this one's, and waiting on it would hold the studio busy for good.
  const active = replacing && replacing.sessionId === sessionId ? replacing : null;
  const outcome = active?.messageId ? cutoutOutcome(session, active.messageId) : null;
  const runs = useRef(0);
  // The reply acted on, so a later render never acts on it twice.
  const handled = useRef<string | null>(null);
  // The replacement still on screen: a step that finishes after the picture
  // changed, or after the studio closed, starts nothing.
  const live = useRef<number | null>(null);
  useEffect(() => {
    live.current = active?.run ?? null;
  });
  useEffect(() => () => {
    live.current = null;
  }, []);

  useEffect(() => {
    const messageId = active?.messageId;
    if (!active || !messageId || !outcome || outcome.state === "waiting") return;
    if (handled.current === messageId) return;
    handled.current = messageId;
    const started = active;
    const step = started.step;
    const settle = () => setReplacing((current) => (current?.run === started.run ? null : current));
    const stillOn = () => live.current === started.run;
    const fail = (message: string) => {
      if (!stillOn()) return;
      settle();
      onError(message);
    };
    // Stopped by the person, so there is nothing to say and nothing more to do.
    if (outcome.state === "stopped") {
      settle();
      return;
    }
    if (outcome.state === "failed") {
      fail(step.stage === "find" ? CUTOUT_FAILED : step.stage === "cut" ? NEW_SUBJECT_FAILED : REMOVAL_FAILED);
      return;
    }
    // Every model step takes its turn the same way, and a refused one ends the replacement.
    const next = (
      following: Step,
      words: string,
      artifactId: string,
      mask?: Parameters<Session["apply"]>[2],
    ) => {
      setReplacing((current) => (current?.run === started.run ? { ...current, step: following, messageId: null } : current));
      apply(
        words,
        artifactId,
        mask,
        following.stage === "remove" ? started.plan.settings : undefined,
        following.stage === "remove" ? started.plan.workflowRevisionId : started.plan.cutout?.workflowRevisionId,
        (accepted: TurnAccepted) =>
          setReplacing((current) =>
            current?.run === started.run ? { ...current, messageId: accepted.assistant_message.id } : current),
        undefined,
        settle,
      );
    };
    if (step.stage === "find") {
      void findOldSubject(outcome.artifactId, started.width, started.height)
        .catch(() => null)
        .then(async (found) => {
          if (!found) return fail(CUTOUT_UNREADABLE);
          if (found === "empty") return fail(NO_SUBJECT_FOUND);
          if (!stillOn()) return;
          const pictureId = await uploadedPicture(started.plan.cutout?.reference).catch(() => null);
          if (!pictureId) return fail(NEW_SUBJECT_FAILED);
          if (!stillOn()) return;
          next({ stage: "cut", ...found }, started.plan.cutout?.words ?? "", pictureId);
        });
      return;
    }
    if (step.stage === "cut") {
      void placeNewSubject(outcome.artifactId, started.width, started.height, step.box)
        .catch(() => null)
        .then((placed) => {
          if (placed === "empty") return fail(NO_NEW_SUBJECT);
          if (!placed) return fail(NEW_SUBJECT_FAILED);
          if (!stillOn()) return;
          const reach = subjectReach(started.width, started.height);
          // Grown by the reach a removed part is given, and softened by half of
          // it, so no ring of the old subject is left and the fill meets the
          // rest gradually.
          next({ stage: "remove", placed }, started.plan.words, started.sourceArtifactId, {
            blob: step.removal,
            featherPx: Math.min(MAX_FEATHER_PX, Math.round(reach / 2)),
            invert: false,
            apply: "blend",
          });
        });
      return;
    }
    localEdit(
      "subject",
      outcome.artifactId,
      () => {
        settle();
        started.onAccepted();
      },
      { subject: { placed: step.placed } },
      // The session already says why: its error shows what the server answered.
      settle,
    );
  }, [active, outcome, apply, localEdit, onError]);

  return {
    busy: active !== null,
    start: (
      plan: StudioApplyPlan,
      sourceArtifactId: string,
      size: { width: number; height: number },
      onAccepted: () => void,
    ) => {
      const cutout = plan.cutout;
      // A subject with nothing to replace it from would only be removed.
      if (!cutout?.reference || !sessionId || active) return;
      runs.current += 1;
      const started: Replacing = {
        run: runs.current,
        sessionId,
        plan,
        sourceArtifactId,
        width: size.width,
        height: size.height,
        onAccepted,
        step: { stage: "find" },
        messageId: null,
      };
      setReplacing(started);
      apply(
        cutout.words,
        sourceArtifactId,
        undefined,
        undefined,
        cutout.workflowRevisionId,
        (accepted) =>
          setReplacing((current) =>
            current?.run === started.run ? { ...current, messageId: accepted.assistant_message.id } : current),
        undefined,
        () => setReplacing((current) => (current?.run === started.run ? null : current)),
      );
    },
  };
}

/** The old subject's box and what it covered, grown by its reach and encoded;
 * "empty" when the cutout holds no subject, or null when it cannot be read. */
async function findOldSubject(
  artifactId: string,
  width: number,
  height: number,
): Promise<{ box: PictureBox; removal: Blob } | "empty" | null> {
  const mask = await readCutoutMask(artifactId, width, height);
  if (!mask) return null;
  const box = solidBox(mask);
  if (!box) return "empty";
  dilate(mask, subjectReach(width, height));
  const removal = await encodeMaskPng(mask);
  return removal ? { box, removal } : null;
}

/** The new subject's picture as an artifact: one the library holds as it is, a chosen file once uploaded. */
async function uploadedPicture(picture: Blob | string | undefined): Promise<string | null> {
  if (typeof picture === "string") return picture;
  if (!picture) return null;
  const file = picture instanceof File ? picture : new File([picture], "studio-subject-picture.png", { type: "image/png" });
  return (await api.upload(file)).id;
}

/** The new subject drawn alone at the picture's size where the old one stood;
 * "empty" when its cutout holds no subject, or null when it cannot be read or drawn. */
async function placeNewSubject(
  artifactId: string,
  width: number,
  height: number,
  box: PictureBox,
): Promise<Blob | "empty" | null> {
  const cutout = await readCutoutPixels(artifactId);
  if (!cutout) return null;
  const placed = placedSubject(cutout, width, height, box);
  if (!placed) return "empty";
  return encodeRgbaPng(width, height, placed);
}
