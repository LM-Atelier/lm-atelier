import { useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { FileJson } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { downloadBytes } from "./format";
import {
  generationRecordFileName,
  missingText,
  omissionText,
  operationText,
  readGenerationRecord,
  removedText,
} from "./generationRecord";

/** Read one output's record as the server wrote it, and show what it holds. */
function GenerationRecordBody({ runId, artifactId }: { runId: string; artifactId: string }) {
  const [includePrompt, setIncludePrompt] = useState(false);
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
  const ready = Boolean(record.data && summary);
  const download = () => {
    if (!record.data || !summary) return;
    // The bytes previewed are the bytes saved, so the file matches its digest.
    downloadBytes(record.data, generationRecordFileName(summary), "application/json");
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
            onChange={(event) => setIncludePrompt(event.target.checked)}
          />
          <span>Include the prompt, and any text the workflow takes as a setting</span>
        </label>
        {record.isPending && <p role="status">Reading the record…</p>}
        {(record.isError || (record.data && !summary)) && (
          <p role="alert">The record could not be made for this output.</p>
        )}
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
          <GenerationRecordBody runId={runId} artifactId={artifactId} />
        </AccessibleDialog>,
        document.body,
      )}
    </>
  );
}
