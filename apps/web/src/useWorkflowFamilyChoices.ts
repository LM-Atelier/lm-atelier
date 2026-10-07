import { useState } from "react";
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { orderFamilies, servesCapability } from "./workflowFamilies";
import { uniqueWorkflowRows } from "./useWorkflowReadPages";
import type { WorkflowSelectorCapability } from "./types";
import type { PageState } from "./WorkflowReadPageControls";

const PAGE_SIZE = 50;

export type WorkflowFamilyBrowseState = {
  search: string;
  setSearch: (value: string) => void;
  pages: PageState;
};

export function useWorkflowFamilyChoices(
  capability: WorkflowSelectorCapability | null,
  selectedId: string | null,
  operation?: string,
) {
  const [search, setSearch] = useState("");
  const query = search.trim();
  const pages = useInfiniteQuery({
    queryKey: ["workflow-families", "choices", capability, operation, query],
    initialPageParam: 0,
    enabled: capability !== null,
    queryFn: ({ pageParam, signal }) => api.workflowFamilies(capability!, false, false, {
      limit: PAGE_SIZE, offset: pageParam, search: query, operation,
      variantLimit: 1, variantCapability: capability!, enabledOnly: true, order: "preference",
    }, signal),
    getNextPageParam: (last, loaded) => last.length === PAGE_SIZE
      ? loaded.reduce((count, page) => count + page.length, 0) : undefined,
    select: data => uniqueWorkflowRows(data.pages.flat(), family => family.id),
  });
  const selected = useQuery({
    queryKey: ["workflow-families", "selected-choice", capability, operation, selectedId],
    enabled: capability !== null && Boolean(selectedId),
    queryFn: async ({ signal }) => (await api.workflowFamilies(capability!, false, false, {
      familyIds: [selectedId!], limit: 1, variantLimit: 1, variantCapability: capability!, operation,
    }, signal)).find(family => family.id === selectedId) ?? null,
  });
  const rows = (pages.data ?? []).filter(family => family.id !== selectedId);
  if (selected.data) rows.push(selected.data);
  const families = capability === null ? []
    : orderFamilies(rows.filter(family => servesCapability(family, capability)), capability);
  const error = selected.error ?? (!query && !pages.data ? pages.error : null);
  return {
    families,
    error,
    isLoading: (selectedId !== null && selected.isLoading) || (!query && pages.isLoading),
    refetch: () => Promise.all([pages.refetch(), ...(selectedId ? [selected.refetch()] : [])]),
    browse: { search, setSearch, pages } satisfies WorkflowFamilyBrowseState,
  };
}
