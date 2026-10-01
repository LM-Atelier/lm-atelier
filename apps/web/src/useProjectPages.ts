import { useInfiniteQuery, useQueries, useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { Project } from "./types";

const PAGE_SIZE = 50;

export function distinctProjects(projects: Project[]) {
  return [...new Map(projects.map((project) => [project.id, project])).values()];
}

export function projectsInSidebarOrder(projects: Project[]) {
  return distinctProjects(projects).sort((left, right) => {
    if (left.pinned !== right.pinned) return left.pinned ? -1 : 1;
    if (left.updated_at !== right.updated_at) return left.updated_at > right.updated_at ? -1 : 1;
    return left.id === right.id ? 0 : left.id > right.id ? -1 : 1;
  });
}

export function useProjectPages(search = "", includeArchived = false) {
  const query = search.trim();
  return useInfiniteQuery({
    queryKey: ["projects", "pages", query, includeArchived],
    initialPageParam: 0,
    queryFn: ({ pageParam, signal }) => api.projects(includeArchived, query, {
      limit: PAGE_SIZE, offset: pageParam, literalSearch: true, signal,
    }),
    getNextPageParam: (lastPage, pages) => lastPage.length === PAGE_SIZE
      ? pages.reduce((count, page) => count + page.length, 0) : undefined,
    select: (data) => distinctProjects(data.pages.flat()),
  });
}

export function useProject(id: string | null | undefined) {
  return useQuery({
    queryKey: ["projects", "detail", id],
    queryFn: ({ signal }) => api.project(id!, signal),
    enabled: Boolean(id),
  });
}

/** Loaded chats can belong to projects outside every visible project page. */
export function useProjectParents(ids: (string | null)[]) {
  const unique = [...new Set(ids.filter((id): id is string => id !== null))].sort();
  const batches: string[][] = [];
  for (let start = 0; start < unique.length; start += 200) batches.push(unique.slice(start, start + 200));
  const queries = useQueries({ queries: batches.map((batch) => ({
    queryKey: ["projects", "parents", batch],
    queryFn: ({ signal }: { signal: AbortSignal }) => api.projects(true, "", { projectIds: batch, limit: 200, signal }),
  })) });
  return {
    data: distinctProjects(queries.flatMap((query) => query.data ?? [])),
    error: queries.find((query) => query.error)?.error,
    isFetching: queries.some((query) => query.isFetching),
    refetch: () => Promise.all(queries.map((query) => query.refetch())),
  };
}
