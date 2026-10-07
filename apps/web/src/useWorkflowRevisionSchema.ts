import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

/** Read only the selected revision; a failed refresh cannot lend stale settings. */
export function useWorkflowRevisionSchema(revisionId: string | null, operation: string | null | undefined) {
  const query = useQuery({
    queryKey: ["workflows", "revision-schema", revisionId],
    queryFn: ({ signal }) => api.workflowRevisionSchema(revisionId!, signal),
    enabled: revisionId !== null && operation !== null,
  });
  const value = query.isError ? undefined : query.data;
  const schema = value && value.revision_id === revisionId
    && (operation === undefined || value.operation === operation)
    ? value.input_schema_json
    : undefined;
  return { schema, isLoading: query.isLoading, error: query.error, retry: query.refetch };
}
