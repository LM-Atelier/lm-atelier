import { useId, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import type { WebSearch } from "./types";
import { SearchQueryFormatting } from "./SearchQueryFormatting";
import "./ChatSearchConsent.css";

type Command = {
  jobId: string;
  revision: number;
} & ({ kind: "edit"; query: string } | {
  kind: "decision"; action: "approve" | "decline" | "cancel";
});

type Props = {
  search: WebSearch;
  onChanged: () => void;
  onUseSource: (url: string) => void;
};

const headings: Record<WebSearch["state"], string> = {
  awaiting_approval: "Search the web?",
  scheduled: "Search starts shortly",
  approved: "Search queued",
  declined: "Continuing without web search",
  cancelled: "Search cancelled",
  dispatching: "Searching the web",
  complete: "Web search sources",
  failed: "Web search could not finish",
  uncertain: "Search was interrupted",
};

export function ChatSearchConsent({ search, onChanged, onUseSource }: Props) {
  const [draft, setDraft] = useState({ revision: search.revision, value: search.query });
  const query = draft.revision === search.revision ? draft.value : search.query;
  const dirty = query !== search.query;
  const queryHelpId = useId();
  const queryFormattingId = useId();
  const invalidQuery = !query.trim() || [...query].length > 2000
    || /[\p{Cc}\p{Cs}\p{Zl}\p{Zp}]/u.test(query);
  const mutation = useMutation({
    mutationFn: (command: Command) => command.kind === "edit"
      ? api.editSearch(command.jobId, command.revision, command.query)
      : api.decideSearch(command.jobId, command.revision, command.action),
    onSuccess: onChanged,
    onError: (error) => {
      if (error instanceof ApiError && error.status === 409) onChanged();
    },
  });
  const actionable = search.job_id !== null && search.revision !== null;
  const waiting = actionable && search.state === "awaiting_approval";
  const cancellable = actionable && ["scheduled", "approved"].includes(search.state);
  const decide = (action: "approve" | "decline" | "cancel") => {
    if (!actionable || mutation.isPending || (action === "approve" && (dirty || invalidQuery))) return;
    mutation.mutate({
      kind: "decision", jobId: search.job_id!, revision: search.revision!, action,
    });
  };
  const save = () => {
    if (!actionable || mutation.isPending || !dirty || invalidQuery) return;
    mutation.mutate({
      kind: "edit", jobId: search.job_id!, revision: search.revision!, query,
    });
  };
  return (
    <section className="chat-search-consent" aria-label="Web search" aria-busy={mutation.isPending}>
      <strong role="status">{headings[search.state]}</strong>
      <p className="chat-search-provider">Provider: {search.provider} at {search.provider_endpoint}</p>
      {waiting ? (
        <label className="chat-search-query">
          Exact query
          <textarea value={query} rows={3} dir="auto"
            aria-invalid={invalidQuery} aria-describedby={[queryHelpId, queryFormattingId].join(" ")}
            onChange={(event) => setDraft({ revision: search.revision, value: event.target.value })}
            readOnly={mutation.isPending} />
        </label>
      ) : <p className="chat-search-query-text" dir="auto">{search.query}</p>}
      <SearchQueryFormatting query={waiting ? query : search.query} descriptionId={queryFormattingId} />
      {search.state === "scheduled" && <p>You can cancel before this query is sent.</p>}
      {waiting && (
        <>
          <p id={queryHelpId} role={invalidQuery ? "alert" : undefined}>
            Use one line of text, without tabs or control characters (1 to 2,000 characters).
          </p>
          <p>The query goes to this provider only after you approve it.</p>
          <div className="chat-search-actions">
            <button className="secondary compact-button" onClick={save}
              aria-disabled={!dirty || invalidQuery || mutation.isPending}>Save query</button>
            <button className="primary compact-button" onClick={() => decide("approve")}
              aria-disabled={dirty || invalidQuery || mutation.isPending}>Search</button>
            <button className="secondary compact-button" onClick={() => decide("decline")}
              aria-disabled={mutation.isPending}>Continue without search</button>
          </div>
          {dirty && <p>Save your changes before approving the query.</p>}
        </>
      )}
      {cancellable && (
        <button className="secondary compact-button" onClick={() => decide("cancel")}
          aria-disabled={mutation.isPending}>Cancel search</button>
      )}
      {mutation.error && <p role="alert">{mutation.error.message}</p>}
      {search.state === "uncertain" && <p>The previous request will not be sent again automatically.</p>}
      {search.results.length > 0 && (
        <ul className="chat-search-results">
          {search.results.map((result) => (
            <li key={result.url}>
              <a href={result.url} target="_blank" rel="noopener noreferrer">{result.title}</a>
              {result.snippet && <p>{result.snippet}</p>}
              <button className="secondary compact-button" onClick={() => onUseSource(result.url)}>
                Add source to message
              </button>
            </li>
          ))}
        </ul>
      )}
      {search.truncated && <p>Some search results were omitted to keep the response concise.</p>}
    </section>
  );
}
