import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { EngineRole } from "./types";

const PAGE_SIZE = 50;

export function uniqueWorkflowRows<T>(rows: T[], key: (row: T) => string): T[] {
  return [...new Map(rows.map(row => [key(row), row])).values()];
}

export function useWorkflowSummaryPages(operation: string | undefined, search: string) {
  const query = search.trim();
  return useInfiniteQuery({
    queryKey: ["workflows", "summaries", "pages", operation, query],
    initialPageParam: 0,
    queryFn: ({ pageParam, signal }) => api.workflowSummaries({
      limit: PAGE_SIZE, offset: pageParam, operation, search: query,
    }, signal),
    getNextPageParam: (last, pages) => last.length === PAGE_SIZE
      ? pages.reduce((count, page) => count + page.length, 0) : undefined,
    select: data => uniqueWorkflowRows(data.pages.flat(), row => row.id),
  });
}

export function useWorkflowSummary(id: string) {
  return useQuery({
    queryKey: ["workflows", "summary", id],
    queryFn: async ({ signal }) => (await api.workflowSummaries({ workflowIds: [id], limit: 1 }, signal))
      .find(row => row.id === id) ?? null,
    enabled: Boolean(id),
  });
}

export function useWorkflowRevisionPages(role: EngineRole, search: string) {
  const query = search.trim();
  return useInfiniteQuery({
    queryKey: ["workflows", "revision-choices", "pages", role, query],
    initialPageParam: 0,
    queryFn: ({ pageParam, signal }) => api.workflowRevisionChoices(signal, {
      limit: PAGE_SIZE, offset: pageParam, role, search: query,
    }),
    getNextPageParam: (last, pages) => last.length === PAGE_SIZE
      ? pages.reduce((count, page) => count + page.length, 0) : undefined,
    select: data => uniqueWorkflowRows(data.pages.flat(), row => row.revision_id),
  });
}

export function useWorkflowRevisionChoice(id: string, role: EngineRole) {
  return useQuery({
    queryKey: ["workflows", "revision-choice", role, id],
    queryFn: async ({ signal }) => (await api.workflowRevisionChoices(signal, { revisionIds: [id], role, limit: 1 }))
      .find(row => row.revision_id === id) ?? null,
    enabled: Boolean(id),
  });
}
