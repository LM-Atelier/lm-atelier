export type PageState = {
  error: Error | null;
  isPending: boolean;
  isFetchingNextPage: boolean;
  isFetchNextPageError: boolean;
  hasNextPage: boolean;
  fetchNextPage: () => Promise<unknown>;
  refetch: () => Promise<unknown>;
};

export function WorkflowReadPageControls({ pages, label }: { pages: PageState; label: string }) {
  return <>
    {pages.isPending && <p role="status">Loading {label}…</p>}
    {pages.error && <div role="alert">{pages.error.message}{" "}
      <button type="button" className="secondary compact-button" onClick={() => {
        if (pages.isFetchNextPageError) void pages.fetchNextPage();
        else void pages.refetch();
      }}>Retry {label}</button>
    </div>}
    {pages.hasNextPage && !pages.error && <button type="button" className="secondary compact-button"
      aria-disabled={pages.isFetchingNextPage} onClick={() => {
        if (!pages.isFetchingNextPage) void pages.fetchNextPage();
      }}>{pages.isFetchingNextPage ? "Loading more " + label + "…" : "Load more " + label}</button>}
  </>;
}
