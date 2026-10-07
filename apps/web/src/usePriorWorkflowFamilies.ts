import { useState } from "react";
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { uniqueWorkflowRows } from "./useWorkflowReadPages";

const PAGE_SIZE = 50;

export function usePriorWorkflowFamilies(selectedId: string | null) {
  const [search, setSearch] = useState("");
  const query = search.trim();
  const pages = useInfiniteQuery({
    queryKey: ["workflow-families", "prior-turn-choices", query],
    initialPageParam: 0,
    queryFn: ({ pageParam, signal }) => api.workflowFamilies(undefined, false, false, {
      limit: PAGE_SIZE, offset: pageParam, search: query, variantLimit: 1, enabledOnly: true,
    }, signal),
    getNextPageParam: (last, loaded) => last.length === PAGE_SIZE
      ? loaded.reduce((count, page) => count + page.length, 0) : undefined,
    select: data => uniqueWorkflowRows(data.pages.flat(), family => family.id),
  });
  const selected = useQuery({
    queryKey: ["workflow-families", "prior-turn-selected", selectedId],
    enabled: Boolean(selectedId),
    queryFn: async ({ signal }) => (await api.workflowFamilies(undefined, false, false, {
      familyIds: [selectedId!], limit: 1, variantLimit: 1,
    }, signal)).find(row => row.id === selectedId) ?? null,
  });
  const rows = (pages.data ?? []).filter(row => row.id !== selectedId);
  if (selected.isSuccess && selected.data) rows.push(selected.data);
  return { families: rows.filter(row => row.enabled && !row.archived), selected,
    browse: { search, setSearch, pages } };
}
