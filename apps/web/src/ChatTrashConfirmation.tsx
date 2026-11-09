import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";
import type { RecoveryCommand, RecoveryImpact } from "./recoveryTypes";

/** How long the warning about a chat's queued or running work stands before it can be confirmed. */
const BUSY_CHAT_WAIT_SECONDS = 3;

type Step = "idle" | "checking" | "cancelling" | "moving";

function holdsWork(impact: RecoveryImpact): boolean {
  return !impact.available_actions.includes("trash") && impact.conflicts.includes("active_work");
}

/** Says that moving the chat cancels its work, and stands a few seconds before that can be confirmed. */
function BusyChatWarning({ revision, waited, onWaited }: {
  revision: string;
  waited: boolean;
  onWaited: (revision: string) => void;
}) {
  const [seconds, setSeconds] = useState(waited ? 0 : BUSY_CHAT_WAIT_SECONDS);
  useEffect(() => {
    if (seconds <= 0) return;
    const timer = window.setTimeout(() => {
      setSeconds(seconds - 1);
      if (seconds === 1) onWaited(revision);
    }, 1_000);
    return () => window.clearTimeout(timer);
  }, [seconds, revision, onWaited]);
  return <>
    <p role="alert">This chat has work queued or running. Moving it to Recently Deleted cancels all of that work first, and restoring the chat later does not bring the work back.</p>
    {seconds > 0 && <p>You can confirm in {seconds} {seconds === 1 ? "second" : "seconds"}.</p>}
  </>;
}

/** Asks before moving a chat to Recently Deleted.
 *
 * The move can be confirmed at once: a confirmation given before the chat's
 * current deletion details arrive waits for them, then moves the chat if they
 * allow it. A chat with work queued or running is never moved by such an early
 * confirmation; it gets its own warning first. Confirming that warning cancels
 * all of the chat's work, then moves the chat as it stands once its work ended.
 */
export function ChatTrashConfirmation({ chatId, title, deleteGeneratedMedia, onConfirm, onCancel }: {
  chatId: string;
  title: string;
  deleteGeneratedMedia: boolean;
  onConfirm: (command: RecoveryCommand) => void | Promise<void>;
  onCancel: () => void;
}) {
  const [operationKey] = useState(() => crypto.randomUUID());
  const [step, setStep] = useState<Step>("idle");
  // The step as it is now, for handlers a render has not caught up with yet: the
  // move can be sent in the moment between the details arriving and the next render.
  const stepNow = useRef<Step>("idle");
  const [waitedRevision, setWaitedRevision] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const submitting = useRef(false);
  const withdrawn = useRef(false);
  const open = useRef(false);
  useEffect(() => {
    open.current = true;
    return () => { open.current = false; };
  }, []);
  const impact = useQuery({
    queryKey: ["chat-deletion-impact", chatId, deleteGeneratedMedia, operationKey],
    queryFn: ({ signal }) => api.deletionImpact(chatId, deleteGeneratedMedia, signal),
    retry: false, refetchOnWindowFocus: false, refetchOnReconnect: false,
  });
  // The details last read stay on show while newer ones load.
  const shown = impact.data;
  const shownBusy = shown !== undefined && holdsWork(shown);
  const settled = shown && !impact.isFetching && !impact.isError ? shown : null;
  const allowed = settled?.available_actions.includes("trash") === true;
  const waited = settled !== null && holdsWork(settled) && waitedRevision === settled.revision;
  const requiresRecheck = error instanceof ApiError && error.status >= 400 && error.status < 500;
  const ready = step === "idle" && !requiresRecheck && !impact.isError
    && (settled ? allowed || waited : !shownBusy);

  const enter = (next: Step) => {
    stepNow.current = next;
    setStep(next);
  };

  const confirm = async () => {
    if (submitting.current) return;
    submitting.current = true;
    withdrawn.current = false;
    setError(null);
    try {
      let current = settled;
      if (current === null) {
        enter("checking");
        const arrived = await impact.refetch({ cancelRefetch: false });
        // A failed check shows its own error; work found only now gets its warning first.
        if (!open.current || withdrawn.current || !arrived.data || arrived.isError) return;
        if (!arrived.data.available_actions.includes("trash")) return;
        current = arrived.data;
      } else if (holdsWork(current)) {
        enter("cancelling");
        await api.cancelAllChatWork(chatId);
        const fresh = await impact.refetch();
        if (!open.current || !fresh.data || fresh.isError) return;
        if (!fresh.data.available_actions.includes("trash")) {
          setError(new Error(holdsWork(fresh.data)
            ? "New work started in this chat while its work was being cancelled."
            : "This chat cannot be moved right now."));
          return;
        }
        current = fresh.data;
      }
      enter("moving");
      await onConfirm({ expected_revision: current.revision, impact_sha256: current.impact_sha256, operation_key: operationKey });
    } catch (failure) {
      if (open.current) setError(failure instanceof Error ? failure : new Error("This chat could not be moved. Try again."));
    } finally {
      submitting.current = false;
      stepNow.current = "idle";
      if (open.current) setStep("idle");
    }
  };

  const confirmLabel = step === "cancelling" ? "Cancelling its work…"
    : step === "checking" || step === "moving" ? "Moving chat…"
    : shownBusy ? "Cancel its work and move to Recently Deleted" : "Move to Recently Deleted";
  return <ConfirmDialog title="Move this chat to Recently Deleted?"
    question={`Move “${title}” and its history to Recently Deleted? You can restore it for 30 days.`}
    tone="action" confirmLabel={confirmLabel} confirmDisabled={!ready}
    onCancel={() => {
      if (submitting.current && stepNow.current !== "checking") return;
      withdrawn.current = true;
      onCancel();
    }}
    onConfirm={() => { if (ready) void confirm(); }} detail={<>
      {impact.isFetching && <p role="status">Checking this chat's current deletion details…</p>}
      {shown && <p>
        {shown.counts.messages} messages · {shown.counts.artifacts} media files
        <br />Retained media: {formatBytes(shown.counts.retained_bytes)} (estimate)
        <br />No media bytes are removed now.
      </p>}
      {deleteGeneratedMedia && <p>At permanent deletion, exclusive generated media also leaves the Media Library. Favorites and media retained elsewhere stay protected.</p>}
      {shownBusy && <BusyChatWarning key={shown.revision} revision={shown.revision}
        waited={waitedRevision === shown.revision} onWaited={setWaitedRevision} />}
      {settled && !allowed && !holdsWork(settled) && <p role="alert">This chat cannot be moved right now.</p>}
      <ErrorCallout message={error?.message ?? impact.error?.message} />
      {(requiresRecheck || impact.isError || (settled && !allowed && !holdsWork(settled))) && !impact.isFetching && step === "idle"
        && <button className="secondary" onClick={() => { setError(null); void impact.refetch(); }}>Check again</button>}
    </>} />;
}
