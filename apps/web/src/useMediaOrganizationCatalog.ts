import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api";
import { parseMediaCatalogPage } from "./mediaOrganization";

export function useMediaOrganizationCatalog<T extends { id: string }>(
  kind: "albums" | "tags", parse: (row: unknown) => T,
) {
  const [query, setQuery] = useState("");
  const [cursors, setCursors] = useState<(string | null)[]>([null]);
  const [epoch, setEpoch] = useState(0);
  const cursor = cursors[cursors.length - 1];
  const result = useQuery({
    queryKey: ["media-organization-catalog-v1", kind, query.trim(), cursor, epoch], retry: false,
    queryFn: async ({ signal }) => parseMediaCatalogPage(await api.mediaOrganizationCatalog(kind, query.trim(), cursor, signal), parse),
  });
  return {
    items: result.data?.items ?? [], query, pending: result.isFetching, failed: result.isError,
    hasPrevious: cursors.length > 1, hasNext: Boolean(result.data?.next_cursor),
    search: (value: string) => { setQuery(value); setCursors([null]); },
    next: () => {
      const next = result.data?.next_cursor;
      if (!result.isFetching && next && !cursors.includes(next)) setCursors([...cursors, next]);
    },
    previous: () => { if (!result.isFetching && cursors.length > 1) setCursors(cursors.slice(0, -1)); },
    refresh: () => { setCursors([null]); setEpoch((value) => value + 1); },
  };
}
