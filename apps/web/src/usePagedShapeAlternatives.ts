import { useState } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";
import { api } from "./api";
import { operationForTurn } from "./turnWorkflow";
import { useShapeAlternatives } from "./shapeAlternatives";
import { uniqueWorkflowRows } from "./useWorkflowReadPages";

export function usePagedShapeAlternatives({
  chatId, capability, hasAttachments, currentRevisionId, enabled,
}: {
  chatId: string | null; capability: "image" | "video" | null; hasAttachments: boolean;
  currentRevisionId: string | null; enabled: boolean;
}) {
  const [search, setSearch] = useState("");
  const query = search.trim();
  const operation = capability ? operationForTurn(capability, hasAttachments) : undefined;
  const available = enabled && Boolean(chatId && capability) && operation !== "image_to_image";
  const pages = useInfiniteQuery({
    queryKey: ["workflow-families", "shape-alternatives", capability, operation, query],
    enabled: available,
    initialPageParam: 0,
    queryFn: ({ pageParam, signal }) => api.workflowFamilies(capability!, false, false, {
      limit: 10, offset: pageParam, variantLimit: 2, search: query, operation,
      readiness: "ready", source: "workflow", enabledOnly: true, order: "preference",
    }, signal),
    getNextPageParam: (last, loaded) => last.length === 10
      ? loaded.reduce((count, page) => count + page.length, 0) : undefined,
    select: data => uniqueWorkflowRows(data.pages.flat(), family => family.id),
  });
  const alternatives = useShapeAlternatives({ chatId, capability, hasAttachments, currentRevisionId,
    enabled: available, families: pages.data ?? [] });
  return alternatives ? { ...alternatives, browse: { search, setSearch, pages } } : undefined;
}
