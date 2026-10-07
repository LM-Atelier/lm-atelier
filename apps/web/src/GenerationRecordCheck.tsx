import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useMutation } from "@tanstack/react-query";
import { FileJson } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { GenerationRecordAdapt } from "./GenerationRecordAdapt";
import { GenerationRecordReplay } from "./GenerationRecordReplay";
import {
  missingText,
  operationText,
  readFileBytes,
  readGenerationRecordCheck,
  readReplayPlan,
  requirementKindText,
  requirementStateText,
  type GenerationRecordCheck as CheckAnswer,
} from "./generationRecord";
import { isEncryptedArchive } from "./projectArchiveFiles";

/** The answer for one record: what it needs, and which of those this computer holds. */
function CheckAnswerView({ answer }: { answer: CheckAnswer }) {
  return (
    <div className="generation-record-body">
      <p>
        {operationText(answer.operation)}.{" "}
        {answer.requirements.length === 0
          ? "This record names no workflow, model, LoRA or input picture to look for."
          : answer.allPresent
            ? "Everything this record names is here and ready."
            : "Some of what this record names is not here or not ready. Nothing was installed or changed."}
      </p>
      {answer.picture && (
        <p>
          The file also holds a copy of the picture, {answer.picture.width} × {answer.picture.height}{" "}
          pixels.
        </p>
      )}
      {answer.requirements.length > 0 && (
        <ul className="generation-record-requirements">
          {answer.requirements.map((item) => (
            <li key={`${item.kind}:${item.sha256}:${item.role ?? ""}`} data-state={item.state}>
              <span>{requirementKindText(item.kind)}</span>
              <code title={item.sha256}>{item.sha256.slice(0, 12)}</code>
              <strong>{requirementStateText(item.state)}</strong>
            </li>
          ))}
        </ul>
      )}
      {answer.missing.length > 0 && (
        <section aria-labelledby={`check-replay-${answer.digest}`}>
          <h3 id={`check-replay-${answer.digest}`}>What the record itself could not vouch for</h3>
          <ul>{answer.missing.map((reason) => <li key={reason}>{missingText(reason)}</li>)}</ul>
        </section>
      )}
    </div>
  );
}

/** Choose a generation record file and see what of it this installation holds.
 *
 * An encrypted record or bundle is recognized by its first bytes and asks for
 * its passphrase; once opened, it is checked exactly as the plain file it holds.
 */
export function GenerationRecordCheck({ onOpenChat }: { onOpenChat?: (chatId: string) => void }) {
  const chooser = useRef<HTMLInputElement>(null);
  const unlockField = useRef<HTMLInputElement>(null);
  const body = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [locked, setLocked] = useState<File | null>(null);
  const [unlock, setUnlock] = useState("");
  const check = useMutation({
    mutationFn: async (source: File | ArrayBuffer) => {
      const content = source instanceof Blob ? await readFileBytes(source) : source;
      const [answer, plan] = await Promise.all([
        api.checkGenerationRecord(content),
        api.planGenerationReplay(content),
      ]);
      return { answer: readGenerationRecordCheck(answer), plan: readReplayPlan(plan), content };
    },
  });
  const opening = useMutation({
    mutationFn: async ({ file, passphrase }: { file: File; passphrase: string }) =>
      api.openEncryptedGenerationRecord(await readFileBytes(file), passphrase),
    // Not kept once finished, so the passphrase it was given does not linger.
    gcTime: 0,
  });
  useEffect(() => {
    if (locked) unlockField.current?.focus();
  }, [locked]);
  const choose = async (file: File) => {
    setOpen(true);
    check.reset();
    opening.reset();
    setUnlock("");
    if (await isEncryptedArchive(file).catch(() => false)) {
      setLocked(file);
      return;
    }
    setLocked(null);
    check.mutate(file);
  };
  const openLocked = () => {
    if (!locked || !unlock || opening.isPending) return;
    opening.mutate({ file: locked, passphrase: unlock }, {
      // Given per call, so it never runs for an open the person has since
      // closed or replaced with another file; a reset detaches it.
      onSuccess: (content) => {
        setLocked(null);
        setUnlock("");
        // The focused form is about to go; focus stays in the dialog.
        body.current?.focus();
        check.mutate(content);
      },
    });
  };
  const close = () => {
    setOpen(false);
    setLocked(null);
    setUnlock("");
    check.reset();
    opening.reset();
  };
  const failure = check.error as { code?: unknown } | null;
  const failureText = failure?.code === "output-recipe-too-large"
    ? "This file is larger than a generation record can be."
    : failure?.code === "output-recipe-bundle-unreadable"
      ? "This file is not a generation record bundle this version can read."
      : "This file is not a generation record this version can read.";

  return (
    <>
      <button
        type="button"
        className="secondary compact-button"
        onClick={() => chooser.current?.click()}
      >
        <FileJson size={16} aria-hidden="true" />Check a record
      </button>
      <input
        ref={chooser}
        type="file"
        accept="application/json,.json,application/zip,.zip,.encrypted,application/octet-stream"
        hidden
        aria-label="Generation record file"
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file) void choose(file);
        }}
      />
      {open && createPortal(
        <AccessibleDialog
          title="Check a generation record"
          eyebrow="What it needs here"
          closeLabel="Close the record check"
          onClose={close}
          className="generation-record-dialog"
        >
          <div ref={body} tabIndex={-1}>
          {locked && (
            <form aria-label="Encrypted record" onSubmit={(event) => { event.preventDefault(); openLocked(); }}>
              <p>{locked.name} is encrypted. Enter its passphrase to check it.</p>
              <label>Record passphrase<input ref={unlockField} type="password" autoComplete="current-password"
                value={unlock} onChange={(event) => setUnlock(event.target.value)} /></label>
              {opening.error && <p role="alert">{opening.error.message}</p>}
              <span className="row-actions">
                <button type="submit" aria-disabled={!unlock || opening.isPending}>
                  {opening.isPending ? "Opening…" : "Open"}
                </button>
              </span>
            </form>
          )}
          {check.isPending && <p role="status">Checking the record…</p>}
          {check.isError && <p role="alert">{failureText}</p>}
          {check.data && <CheckAnswerView answer={check.data.answer} />}
          {check.data && onOpenChat && (
            <GenerationRecordReplay
              plan={check.data.plan}
              content={check.data.content}
              onStarted={(chatId) => {
                close();
                onOpenChat(chatId);
              }}
            />
          )}
          {check.data && onOpenChat && (
            <GenerationRecordAdapt
              plan={check.data.plan}
              requirements={check.data.answer.requirements}
              bundledInputs={check.data.answer.bundledInputs}
              content={check.data.content}
              onStarted={(chatId) => {
                close();
                onOpenChat(chatId);
              }}
            />
          )}
          </div>
        </AccessibleDialog>,
        document.body,
      )}
    </>
  );
}
