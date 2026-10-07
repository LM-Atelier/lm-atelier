import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { uniqueWorkflowRows } from "./useWorkflowReadPages";
import type { WorkflowFamily, WorkflowVariantReadiness } from "./types";

export type WorkflowLibraryFilters = {
  search: string;
  operation: string;
  readiness: "" | WorkflowVariantReadiness;
  source: "" | "profile" | "workflow";
  order: "name" | "readiness";
  defaultsOnly: boolean;
};
export const EMPTY_LIBRARY_FILTERS: WorkflowLibraryFilters = {
  search: "", operation: "", readiness: "", source: "", order: "name", defaultsOnly: false,
};
const FAMILY_PAGE_SIZE = 10;
const VARIANT_PAGE_SIZE = 5;
const SUMMARY_PAGE_SIZE = 20;

export function useWorkflowLibraryReads(filters: WorkflowLibraryFilters, includeArchived: boolean) {
  const options = {
    ...filters, search: filters.search.trim(), operation: filters.operation || undefined,
    readiness: filters.readiness || undefined, source: filters.source || undefined,
  };
  const families = useInfiniteQuery({
    queryKey: ["workflow-families", "library-pages", includeArchived, options],
    initialPageParam: 0,
    queryFn: ({ pageParam, signal }) => api.workflowFamilies(undefined, includeArchived, true, {
      ...options, limit: FAMILY_PAGE_SIZE, offset: pageParam, variantLimit: VARIANT_PAGE_SIZE,
    }, signal),
    getNextPageParam: (last, pages) => last.length === FAMILY_PAGE_SIZE
      ? pages.reduce((total, page) => total + page.length, 0) : undefined,
    select: data => uniqueWorkflowRows(data.pages.flat(), family => family.id),
  });
  const showUngrouped = !filters.source && !filters.readiness && !filters.defaultsOnly;
  const ungrouped = useInfiniteQuery({
    queryKey: ["workflows", "summaries", "ungrouped", options.search, options.operation],
    enabled: showUngrouped,
    initialPageParam: 0,
    queryFn: ({ pageParam, signal }) => api.workflowSummaries({
      limit: SUMMARY_PAGE_SIZE, offset: pageParam, search: options.search,
      operation: options.operation, ungroupedOnly: true,
    }, signal),
    getNextPageParam: (last, pages) => last.length === SUMMARY_PAGE_SIZE
      ? pages.reduce((total, page) => total + page.length, 0) : undefined,
    select: data => uniqueWorkflowRows(data.pages.flat(), row => row.id),
  });
  const operations = useQuery({
    queryKey: ["workflow-families", "operations", includeArchived],
    queryFn: ({ signal }) => api.workflowFamilyOperations(includeArchived, signal),
  });
  return { families, ungrouped, operations, showUngrouped };
}

export function useWorkflowFamilyVariantPages(
  familyId: string, operation?: string, readiness?: WorkflowVariantReadiness,
  initial?: WorkflowFamily, enabled = true,
) {
  return useInfiniteQuery({
    queryKey: ["workflow-families", "variant-pages", familyId, operation, readiness],
    enabled,
    initialPageParam: 0,
    initialData: initial ? { pages: [initial], pageParams: [0] } : undefined,
    queryFn: async ({ pageParam, signal }) => (await api.workflowFamilies(undefined, true, false, {
      familyIds: [familyId], limit: 1, variantLimit: VARIANT_PAGE_SIZE, variantOffset: pageParam,
      operation, readiness,
    }, signal)).find(family => family.id === familyId) ?? null,
    getNextPageParam: (last, pages) => {
      if (!last) return undefined;
      const count = pages.reduce((total, page) => total + (page?.variants.length ?? 0), 0);
      return last.variant_count != null ? count < last.variant_count ? count : undefined
        : last.variants.length === VARIANT_PAGE_SIZE ? count : undefined;
    },
    select: data => uniqueWorkflowRows(data.pages.flatMap(page => page?.variants ?? []), row => row.id),
  });
}

export function useSelectedWorkflowFamily(workflowId: string | null) {
  return useQuery({
    queryKey: ["workflow-families", "selected-variant", workflowId],
    enabled: workflowId !== null,
    queryFn: async ({ signal }) => (await api.workflowFamilies(undefined, true, false, {
      workflowIds: [workflowId!], limit: 1, variantLimit: 1,
    }, signal)).find(family => family.variants.some(variant => variant.id === workflowId)) ?? null,
  });
}
