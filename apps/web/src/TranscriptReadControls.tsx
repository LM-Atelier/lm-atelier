import type { TranscriptReads } from "./useChatTranscript";

export function TranscriptReadFailure({ reads }: { reads?: TranscriptReads }) {
  if (reads?.error) return <div role="alert"><p>{reads.error.message}</p>
    <button onClick={reads.retry}>Retry conversation</button></div>;
  return reads?.loading ? <p role="status">Loading conversation…</p> : null;
}

export function TranscriptReadControls({ reads, onOlder }: {
  reads: TranscriptReads; onOlder: () => void;
}) {
  return <>
    {reads.searchError && <div role="alert"><p>{reads.searchError.message}</p>
      <button onClick={reads.retrySearches}>Retry searches</button></div>}
    {(reads.hasMorePending || reads.hasLoadedPending) && <button aria-disabled={!reads.hasMorePending || reads.loadingPending}
      onClick={() => { if (reads.hasMorePending && !reads.loadingPending) reads.loadPending(); }}>
      {reads.loadingPending ? "Loading pending searches…"
        : reads.hasMorePending ? "Load more pending searches" : "All pending searches loaded"}
    </button>}
    {reads.olderError && <p role="alert">{reads.olderError.message}</p>}
    <button aria-disabled={!reads.hasOlder || reads.loadingOlder}
      onClick={() => { if (reads.hasOlder && !reads.loadingOlder) onOlder(); }}>
      {reads.loadingOlder ? "Loading older messages…" : reads.hasOlder ? "Load older messages" : "All messages loaded"}
    </button>
  </>;
}
