import { useState } from "react";
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { uniqueWorkflowRows } from "./useWorkflowReadPages";

const PAGE_SIZE = 50;

/** Ready workflows to choose from: by default those that make a picture from words. */
export function useReadyWorkflowChoices(
  selectedIds: string[],
  operation: "text_to_image" | "image_to_image" | "text_to_video" = "text_to_image",
) {
  // Sent only for a video, so a picture's workflows are asked for exactly as before.
  const made = operation === "text_to_video" ? { selectorCapability: "video" as const } : {};
  const [search, setSearch] = useState("");
  const query = search.trim();
  const ids = [...new Set(selectedIds.filter(Boolean))].sort();
  const pages = useInfiniteQuery({
    queryKey: ["workflow-families", "ready-revisions", query, operation],
    initialPageParam: 0,
    queryFn: ({ pageParam, signal }) => api.workflowReadyRevisions({
      operation, limit: PAGE_SIZE, offset: pageParam, search: query, ...made,
    }, signal),
    getNextPageParam: (last, loaded) => last.length === PAGE_SIZE
      ? loaded.reduce((count, page) => count + page.length, 0) : undefined,
    select: data => uniqueWorkflowRows(data.pages.flat(), row => row.revision_id),
  });
  const selected = useQuery({
    queryKey: ["workflow-families", "selected-ready-revisions", ids, operation],
    enabled: ids.length > 0,
    queryFn: ({ signal }) => api.workflowReadyRevisions({
      operation, limit: 200, revisionIds: ids, ...made,
    }, signal),
  });
  const rows = (pages.data ?? []).filter(row => !ids.includes(row.revision_id));
  if (selected.isSuccess) rows.push(...selected.data.filter(row => ids.includes(row.revision_id)));
  return {
    rows, search, setSearch, pages, selected,
    hasSelection: ids.length > 0,
    missingLabel: selected.isSuccess
      ? "Previously selected workflow (currently unavailable)" : "Previously selected workflow",
  };
}
