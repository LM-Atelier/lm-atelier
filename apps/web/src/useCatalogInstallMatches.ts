import { useQueries } from "@tanstack/react-query";
import { api } from "./api";
import type { CatalogModel } from "./types";

export function useCatalogInstallMatches(items: CatalogModel[], role: string) {
  const batches: CatalogModel[][] = [];
  if (role === "chat" || role === "image" || role === "video") {
    for (let offset = 0; offset < items.length; offset += 200) batches.push(items.slice(offset, offset + 200));
  }
  const queries = useQueries({ queries: batches.map((batch) => {
    const remoteIds = [...new Set(batch.map((model) => model.remote_id))];
    const workflowTemplateIds = [...new Set(batch.flatMap((model) => model.workflow_template_id ? [model.workflow_template_id] : []))];
    return {
      queryKey: ["models", "catalog-matches", role, remoteIds, workflowTemplateIds],
      queryFn: () => api.catalogInstallMatches({ role: role as "chat" | "image" | "video", remoteIds, workflowTemplateIds }),
    };
  }) });
  const statusFor = (model: CatalogModel): "checking" | "unavailable" | "installed" | "idle" => {
    const index = items.indexOf(model);
    const query = index < 0 ? undefined : queries[Math.floor(index / 200)];
    if (query?.isError) return "unavailable";
    if (!query?.data) return "checking";
    const installed = model.workflow_template_id
      ? query.data.workflow_template_ids.includes(model.workflow_template_id)
      : query.data.remote_ids.includes(model.remote_id);
    return installed ? "installed" : "idle";
  };
  return {
    statusFor,
    error: queries.find((query) => query.isError)?.error,
    retry: () => { for (const query of queries) if (query.isError && !query.isFetching) void query.refetch(); },
  };
}
