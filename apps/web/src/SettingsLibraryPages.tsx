import { useInfiniteQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { useState, type ReactNode } from "react";
import { ErrorCallout } from "./ErrorCallout";

export function SettingsLibraryPages<T>({ resource, label, fetchPage, children }: {
  resource: "profiles" | "presets";
  label: string;
  fetchPage: (options: { limit: number; offset: number; search: string }) => Promise<T[]>;
  children: (rows: T[]) => ReactNode;
}) {
  const [search, setSearch] = useState("");
  const query = useInfiniteQuery({
    queryKey: [resource, "settings-library", search],
    queryFn: ({ pageParam }) => fetchPage({ limit: 50, offset: pageParam, search }),
    initialPageParam: 0,
    getNextPageParam: (lastPage, _pages, offset) => lastPage.length === 50 ? offset + 50 : undefined,
    retry: false,
  });
  const rows = query.data?.pages.flat() ?? [];
  return <>
    <div className="search-box">
      <Search size={17} aria-hidden="true" />
      <input type="search" aria-label={`Search ${label}`} placeholder={`Search ${label}`}
        maxLength={500} value={search} onChange={(event) => setSearch(event.target.value)} />
    </div>
    {query.isPending && <p role="status">Loading {label}...</p>}
    {query.isSuccess && !query.isFetching && rows.length === 0 && <p>No {label} found.</p>}
    {children(rows)}
    {query.error && <>
      <ErrorCallout message={query.error.message} />
      <button type="button" className="secondary compact-button" aria-disabled={query.isFetching}
        onClick={() => { if (!query.isFetching) void (query.isFetchNextPageError ? query.fetchNextPage() : query.refetch()); }}
      >Retry {label}</button>
    </>}
    {query.hasNextPage && !query.isError && <div className="load-more">
      <button type="button" className="secondary compact-button" aria-disabled={query.isFetching}
        onClick={() => { if (!query.isFetching) void query.fetchNextPage(); }}
      >{query.isFetchingNextPage ? `Loading ${label}...` : `More ${label}`}</button>
    </div>}
  </>;
}
