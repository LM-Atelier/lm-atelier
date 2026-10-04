import { useEffect, useId, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { FileJson } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { downloadBytes, formatBytes } from "./format";
import {
  generationRecordBundleFileName,
  generationRecordFileName,
  missingText,
  omissionText,
  operationText,
  readGenerationRecord,
  RECIPE_DRAFT_OPERATIONS,
  readReplayOutcome,
  removedText,
  replayOutcomeText,
} from "./generationRecord";
import { GenerationEditRecipeDialog } from "./GenerationEditRecipeDialog";
import { RecipeDraftDialog } from "./RecipeDraftDialog";

/** What the person is told when the picture could not be saved with its record. */
const PICTURE_PROBLEMS: Record<string, string> = {
  "output-recipe-changed": "The record changed since it was shown. Check it again, then save.",
  "output-recipe-bundle-too-large":
    "This picture is too large to save with its record. The record can still be saved on its own.",
  "output-recipe-bundle-input-missing":
    "One of its input pictures is no longer here, so they cannot be saved with it. Save it without them.",
  "output-recipe-bundle-input-unreadable":
    "One of its input pictures is missing or changed, so they cannot be saved with it. Save it without them.",
  "output-recipe-bundle-input-uncopyable":
    "One of its input pictures cannot be copied into the file. Save it without them.",
  "output-recipe-bundle-too-many-inputs":
    "It has more input pictures than can be saved with it. Save it without them.",
};

/** What the person is told when a file could not be saved encrypted. */
const SEAL_PROBLEMS: Record<string, string> = {
  "output-recipe-changed": "The record changed since it was shown. Check it again, then save.",
  "archive-passphrase-invalid": "That passphrase is too long. Use at most 1024 bytes.",
  "archive-key-derivation-failed":
    "This computer could not set aside the memory the passphrase needs. Close other applications and try again.",
  "output-recipe-encryption-unverified":
    "The encrypted file could not be checked after it was written, so it was not saved.",
};

/** Said instead when inputs were asked for: the inputs, or the picture alone, may be what does not fit. */
const TOO_LARGE_WITH_INPUTS =
  "The pictures are too large to save together with the record. Try again without its inputs; if this picture alone is too large, save the record on its own.";

/** Read one output's record as the server wrote it, and show what it holds. */
function GenerationRecordBody({
  runId,
  artifactId,
  kind,
  onKeepRecipe,
}: {
  runId: string;
  artifactId: string;
  kind: "image" | "video";
  /** Offered for a generation whose settings a recipe can hold, and for an edit as a Studio recipe. */
  onKeepRecipe: (kind: "generation" | "edit") => void;
}) {
  const [includePrompt, setIncludePrompt] = useState(false);
  const [includeInputs, setIncludeInputs] = useState(false);
  const [savingPicture, setSavingPicture] = useState(false);
  const [pictureProblem, setPictureProblem] = useState<string | null>(null);
  const [encrypt, setEncrypt] = useState(false);
  const [passphrase, setPassphrase] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [savingRecord, setSavingRecord] = useState(false);
  const [recordProblem, setRecordProblem] = useState<string | null>(null);
  const hint = useId();
  // Typed twice, so a slip of the keyboard does not lock the file for good.
  const sealable = !encrypt || (passphrase.length > 0 && passphrase === confirmation);
  const saving = savingPicture || savingRecord;
  // A copy still being made when the dialog closes is stopped, so nothing is
  // saved after the person has left.
  const [pictureRequests] = useState(() => new Set<AbortController>());
  useEffect(() => () => {
    for (const request of pictureRequests) request.abort();
  }, [pictureRequests]);
  const record = useQuery({
    queryKey: ["generation-record", runId, artifactId, includePrompt],
    queryFn: ({ signal }) => api.generationRecord(runId, artifactId, includePrompt, signal),
    retry: false,
    staleTime: 0,
    gcTime: 0,
  });
  const summary = useMemo(() => {
    if (!record.data) return null;
    try {
      return readGenerationRecord(record.data);
    } catch {
      return null;
    }
  }, [record.data]);
  // Only a run generated again from a record has an answer; any other is simply not shown.
  const replayed = useQuery({
    queryKey: ["replay-result", runId],
    queryFn: ({ signal }) => api.replayResult(runId, signal),
    retry: false,
    staleTime: 0,
    gcTime: 0,
  });
  const outcome = useMemo(() => {
    try {
      return replayed.data ? readReplayOutcome(replayed.data) : null;
    } catch {
      return null;
    }
  }, [replayed.data]);
  const ready = Boolean(record.data && summary);
  const download = async () => {
    if (!record.data || !summary || !sealable || savingRecord) return;
    if (!encrypt) {
      // The bytes previewed are the bytes saved, so the file matches its digest.
      downloadBytes(record.data, generationRecordFileName(summary), "application/json");
      return;
    }
    const request = new AbortController();
    pictureRequests.add(request);
    setSavingRecord(true);
    setRecordProblem(null);
    try {
      // Encrypted from the record named by the digest on screen, so it holds no other.
      const bytes = await api.encryptedGenerationRecord(
        runId, artifactId, includePrompt, summary.digest, passphrase, request.signal,
      );
      if (!request.signal.aborted) {
        downloadBytes(bytes, `${generationRecordFileName(summary)}.encrypted`, "application/octet-stream");
      }
    } catch (error) {
      if (request.signal.aborted) return;
      const code = (error as { code?: unknown } | null)?.code;
      setRecordProblem((typeof code === "string" ? SEAL_PROBLEMS[code] : undefined) ?? "The record could not be saved encrypted.");
      if (code === "output-recipe-changed") void record.refetch();
    } finally {
      pictureRequests.delete(request);
      if (!request.signal.aborted) setSavingRecord(false);
    }
  };
  const downloadWithPicture = async () => {
    if (!summary || savingPicture || !sealable) return;
    const request = new AbortController();
    pictureRequests.add(request);
    setSavingPicture(true);
    setPictureProblem(null);
    try {
      // The digest names the record on screen, so the file never holds another.
      // Only offered, and only sent, when the record names some.
      const inputs = includeInputs && summary.inputCount > 0;
      const bytes = encrypt
        ? await api.encryptedGenerationRecordBundle(
          runId, artifactId, includePrompt, summary.digest, inputs, passphrase, request.signal,
        )
        : await api.generationRecordBundle(runId, artifactId, includePrompt, summary.digest, inputs, request.signal);
      if (!request.signal.aborted) {
        const name = generationRecordBundleFileName(summary);
        downloadBytes(bytes, encrypt ? `${name}.encrypted` : name, encrypt ? "application/octet-stream" : "application/zip");
      }
    } catch (error) {
      if (request.signal.aborted) return;
      const code = (error as { code?: unknown } | null)?.code;
      const known = code === "output-recipe-bundle-too-large" && includeInputs
        ? TOO_LARGE_WITH_INPUTS
        : typeof code === "string" ? PICTURE_PROBLEMS[code] ?? SEAL_PROBLEMS[code] : undefined;
      setPictureProblem(known ?? "The picture could not be saved with its record.");
      if (code === "output-recipe-changed") void record.refetch();
    } finally {
      pictureRequests.delete(request);
      if (!request.signal.aborted) setSavingPicture(false);
    }
  };

  return (
    <>
      <div className="generation-record-body">
        <p>
          A record of how this was made, to keep or share. It names the output, model, workflow
          and inputs by their file hashes and never by anything that only means something on this
          computer.
        </p>
        <label className="generation-record-choice">
          <input
            type="checkbox"
            checked={includePrompt}
            // The choice is part of the file being made; it waits until that is saved.
            disabled={saving}
            onChange={(event) => {
              setIncludePrompt(event.target.checked);
              setPictureProblem(null);
            }}
          />
          <span>Include the prompt, and any text the workflow takes as a setting</span>
        </label>
        {kind === "image" && (
          <p>
            Saved with the picture, the file also holds a copy of it with only its pixels. Nothing
            else written inside the picture file, such as the workflow that made it, is copied, and
            a color profile is applied to the pixels rather than kept.
          </p>
        )}
        {/* Not offered when the record already says one of them is gone, since saving them would fail. */}
        {kind === "image" && summary && summary.inputCount > 0 && !summary.missing.includes("input_unavailable") && (
          <label className="generation-record-choice">
            <input
              type="checkbox"
              checked={includeInputs}
              disabled={saving}
              onChange={(event) => {
                setIncludeInputs(event.target.checked);
                setPictureProblem(null);
              }}
            />
            <span>{inputChoiceText(summary.inputCount, summary.inputBytes)}</span>
          </label>
        )}
        <label className="generation-record-choice">
          <input
            type="checkbox"
            checked={encrypt}
            disabled={saving}
            onChange={(event) => {
              setEncrypt(event.target.checked);
              setPictureProblem(null);
              setRecordProblem(null);
            }}
          />
          <span>
            Encrypt the saved file with a passphrase. It cannot be opened without it, and a forgotten
            passphrase cannot be recovered.
          </span>
        </label>
        {encrypt && (
          <span className="row-actions">
            <label>Passphrase<input type="password" autoComplete="new-password" value={passphrase}
              readOnly={saving} onChange={(event) => setPassphrase(event.target.value)} /></label>
            <label>Confirm passphrase<input type="password" autoComplete="new-password" value={confirmation}
              readOnly={saving} onChange={(event) => setConfirmation(event.target.value)} /></label>
          </span>
        )}
        {encrypt && !sealable && (
          <p className="muted" id={hint}>
            {confirmation.length > 0 && passphrase !== confirmation
              ? "The passphrases do not match."
              : "Type the passphrase twice to save the file encrypted."}
          </p>
        )}
        {outcome && <p>{replayOutcomeText(outcome)}</p>}
        {record.isPending && <p role="status">Reading the record…</p>}
        {(record.isError || (record.data && !summary)) && (
          <p role="alert">The record could not be made for this output.</p>
        )}
        {savingPicture && <p role="status">Copying the picture…</p>}
        {savingRecord && <p role="status">Encrypting the record…</p>}
        {pictureProblem && <p role="alert">{pictureProblem}</p>}
        {recordProblem && <p role="alert">{recordProblem}</p>}
        {summary && (
          <>
            <dl className="generation-record-facts">
              <dt>Made by</dt>
              <dd>{operationText(summary.operation)}</dd>
              <dt>Prompt</dt>
              <dd>{summary.promptIncluded ? "Included" : omissionText(summary.promptOmittedReason)}</dd>
              <dt>Seed</dt>
              <dd>{summary.seed === null ? "Not recorded" : summary.seed}</dd>
              <dt>Settings</dt>
              <dd>{summary.settingCount}</dd>
              <dt>Inputs</dt>
              <dd>{summary.inputCount}</dd>
              <dt>Workflow</dt>
              <dd>
                {summary.workflowVerified === null
                  ? "Not available"
                  : summary.workflowVerified ? "Identified" : "Not confirmed"}
              </dd>
              <dt>Model files</dt>
              <dd>{summary.modelFileCount ?? "Not recorded"}</dd>
              <dt>Added LoRAs</dt>
              <dd>{summary.loraCount}</dd>
            </dl>
            {summary.removed.length > 0 && (
              <section aria-labelledby={`left-out-${summary.outputSha256}`}>
                <h3 id={`left-out-${summary.outputSha256}`}>Left out</h3>
                <ul>{summary.removed.map((name) => <li key={name}>{removedText(name)}</li>)}</ul>
              </section>
            )}
            {summary.missing.length > 0 && (
              <section aria-labelledby={`replay-${summary.outputSha256}`}>
                <h3 id={`replay-${summary.outputSha256}`}>Not enough to remake it exactly</h3>
                <ul>{summary.missing.map((reason) => <li key={reason}>{missingText(reason)}</li>)}</ul>
              </section>
            )}
          </>
        )}
      </div>
      <footer>
        {summary && (RECIPE_DRAFT_OPERATIONS.has(summary.operation) || summary.operation === "image_to_image") && (
          <button type="button" className="secondary" aria-disabled={saving} onClick={() => {
            if (!saving) onKeepRecipe(summary.operation === "image_to_image" ? "edit" : "generation");
          }}>
            Keep as a recipe
          </button>
        )}
        {kind === "image" && (
          <button
            type="button"
            className="secondary"
            aria-disabled={!ready || saving || !sealable}
            aria-describedby={sealable ? undefined : hint}
            onClick={() => {
              if (ready && !saving && sealable) void downloadWithPicture();
            }}
          >
            Download with picture
          </button>
        )}
        <button
          type="button"
          className="primary"
          aria-disabled={!ready || saving || !sealable}
          aria-describedby={sealable ? undefined : hint}
          onClick={() => {
            if (ready && !saving && sealable) void download();
          }}
        >
          Download record
        </button>
      </footer>
    </>
  );
}

/** The inputs choice, saying how many pictures it adds and about how much they hold as stored. */
function inputChoiceText(count: number, bytes: number | null): string {
  const pictures = count === 1 ? "its input picture" : `its ${count} input pictures`;
  const size = bytes === null ? "" : `, about ${formatBytes(bytes)} as stored`;
  return `Also save ${pictures} with it, copied the same way${size}`;
}

/** Keep the settings one generation ran with as a recipe, reviewed before it is saved. */
function GenerationRecipeDialog({ runId, onClose }: { runId: string; onClose: () => void }) {
  const draft = useQuery({
    queryKey: ["output-recipe-draft", runId],
    queryFn: ({ signal }) => api.outputRecipeDraft(runId, signal),
    retry: false,
    staleTime: 0,
    gcTime: 0,
  });
  return (
    <RecipeDraftDialog
      title="Keep these settings as a recipe"
      eyebrow="Recipe from a generation"
      draft={draft}
      failure={(error) => error.message || "A recipe could not be drafted from this generation."}
      onClose={onClose}
    />
  );
}

/** Open the record of one generated picture or video, and save it as a file. */
export function GenerationRecordButton({
  runId,
  artifactId,
  kind,
}: {
  runId: string;
  artifactId: string;
  kind: "image" | "video";
}) {
  const [open, setOpen] = useState<"record" | "generation" | "edit" | null>(null);
  return (
    <>
      <button
        type="button"
        className="icon-button"
        aria-label={`Generation record of this ${kind}`}
        title="Generation record"
        onClick={() => setOpen("record")}
      >
        <FileJson size={14} aria-hidden="true" />
      </button>
      {/* Outside the caption it is opened from, so the caption's own styles
          for its small controls never reach the dialog's buttons. */}
      {open === "record" && createPortal(
        <AccessibleDialog
          title="Generation record"
          eyebrow="How this was made"
          closeLabel="Close generation record"
          onClose={() => setOpen(null)}
          className="generation-record-dialog"
        >
          <GenerationRecordBody runId={runId} artifactId={artifactId} kind={kind}
            onKeepRecipe={setOpen} />
        </AccessibleDialog>,
        document.body,
      )}
      {open === "generation" && createPortal(
        <GenerationRecipeDialog runId={runId} onClose={() => setOpen(null)} />,
        document.body,
      )}
      {open === "edit" && createPortal(
        <GenerationEditRecipeDialog runId={runId} onClose={() => setOpen(null)} />,
        document.body,
      )}
    </>
  );
}
