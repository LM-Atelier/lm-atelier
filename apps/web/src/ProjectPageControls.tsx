import type { useProjectPages } from "./useProjectPages";

export function ProjectPageControls({ pages }: { pages: ReturnType<typeof useProjectPages> }) {
  return <>
    {pages.isPending && <p role="status">Loading projects…</p>}
    {pages.error && <p role="alert">{pages.error.message} <button type="button" onClick={() => void (
      pages.isFetchNextPageError ? pages.fetchNextPage() : pages.refetch()
    )}>Retry projects</button></p>}
    {pages.hasNextPage && <button type="button" className="secondary compact-button" aria-disabled={pages.isFetching}
      onClick={() => { if (!pages.isFetching) void pages.fetchNextPage(); }}>
      {pages.isFetchingNextPage ? "Loading projects…" : "Load more projects"}
    </button>}
  </>;
}
