import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { FileJson } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { downloadBytes } from "./format";
import {
  generationRecordBundleFileName,
  generationRecordFileName,
  missingText,
  omissionText,
  operationText,
  readGenerationRecord,
  readReplayOutcome,
  removedText,
  replayOutcomeText,
} from "./generationRecord";

/** What the person is told when the picture could not be saved with its record. */
const PICTURE_PROBLEMS: Record<string, string> = {
  "output-recipe-changed": "The record changed since it was shown. Check it again, then save.",
  "output-recipe-bundle-too-large":
    "This picture is too large to save with its record. The record can still be saved on its own.",
};

/** Read one output's record as the server wrote it, and show what it holds. */
function GenerationRecordBody({
  runId,
  artifactId,
  kind,
}: {
  runId: string;
  artifactId: string;
  kind: "image" | "video";
}) {
  const [includePrompt, setIncludePrompt] = useState(false);
  const [savingPicture, setSavingPicture] = useState(false);
  const [pictureProblem, setPictureProblem] = useState<string | null>(null);
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
  const download = () => {
    if (!record.data || !summary) return;
    // The bytes previewed are the bytes saved, so the file matches its digest.
    downloadBytes(record.data, generationRecordFileName(summary), "application/json");
  };
  const downloadWithPicture = async () => {
    if (!summary || savingPicture) return;
    const request = new AbortController();
    pictureRequests.add(request);
    setSavingPicture(true);
    setPictureProblem(null);
    try {
      // The digest names the record on screen, so the file never holds another.
      const bytes = await api.generationRecordBundle(
        runId,
        artifactId,
        includePrompt,
        summary.digest,
        request.signal,
      );
      if (!request.signal.aborted) {
        downloadBytes(bytes, generationRecordBundleFileName(summary), "application/zip");
      }
    } catch (error) {
      if (request.signal.aborted) return;
      const code = (error as { code?: unknown } | null)?.code;
      const known = typeof code === "string" ? PICTURE_PROBLEMS[code] : undefined;
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
            disabled={savingPicture}
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
        {outcome && <p>{replayOutcomeText(outcome)}</p>}
        {record.isPending && <p role="status">Reading the record…</p>}
        {(record.isError || (record.data && !summary)) && (
          <p role="alert">The record could not be made for this output.</p>
        )}
        {savingPicture && <p role="status">Copying the picture…</p>}
        {pictureProblem && <p role="alert">{pictureProblem}</p>}
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
        {kind === "image" && (
          <button
            type="button"
            className="secondary"
            aria-disabled={!ready || savingPicture}
            onClick={() => {
              if (ready && !savingPicture) void downloadWithPicture();
            }}
          >
            Download with picture
          </button>
        )}
        <button
          type="button"
          className="primary"
          aria-disabled={!ready}
          onClick={() => {
            if (ready) download();
          }}
        >
          Download record
        </button>
      </footer>
    </>
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
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        className="icon-button"
        aria-label={`Generation record of this ${kind}`}
        title="Generation record"
        onClick={() => setOpen(true)}
      >
        <FileJson size={14} aria-hidden="true" />
      </button>
      {/* Outside the caption it is opened from, so the caption's own styles
          for its small controls never reach the dialog's buttons. */}
      {open && createPortal(
        <AccessibleDialog
          title="Generation record"
          eyebrow="How this was made"
          closeLabel="Close generation record"
          onClose={() => setOpen(false)}
          className="generation-record-dialog"
        >
          <GenerationRecordBody runId={runId} artifactId={artifactId} kind={kind} />
        </AccessibleDialog>,
        document.body,
      )}
    </>
  );
}
