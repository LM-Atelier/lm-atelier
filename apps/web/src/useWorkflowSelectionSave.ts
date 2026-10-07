import { useIsMutating, useMutation, useQueryClient } from "@tanstack/react-query";
import type { MutationFilters } from "@tanstack/react-query";
import { api } from "./api";
import type {
  ChatWorkflowSelectionInput, ProjectWorkflowSelectionInput, WorkflowSelectorCapability,
} from "./types";

type Scope = { kind: "chat" | "project"; id: string; capability: WorkflowSelectorCapability };
type Selection = ChatWorkflowSelectionInput | ProjectWorkflowSelectionInput;
type Save = Scope & { selection: Selection };

/** Keep a workflow choice pending across every control that can change its scope. */
export function useWorkflowSelectionSave(scope: Scope | null) {
  const client = useQueryClient();
  const matches = (variables: unknown) => typeof variables === "object" && variables !== null
    && scope !== null && "kind" in variables && variables.kind === scope.kind
    && "id" in variables && variables.id === scope.id
    && "capability" in variables && variables.capability === scope.capability;
  const filter: MutationFilters = {
    mutationKey: ["workflow-selection-save"], exact: true,
    predicate: (mutation) => matches(mutation.state.variables),
  };
  const saving = useIsMutating(filter) > 0;
  const save = useMutation({
    mutationKey: filter.mutationKey,
    mutationFn: ({ kind, id, capability, selection }: Save) => kind === "chat"
      ? api.setChatWorkflowSelection(id, capability, selection as ChatWorkflowSelectionInput)
      : api.setProjectWorkflowSelection(id, capability, selection as ProjectWorkflowSelectionInput),
    onSuccess: (_saved, { kind, id }) => client.invalidateQueries({
      queryKey: [kind, id, "workflow-selections"],
    }),
  });
  return {
    saving,
    error: matches(save.variables) ? save.error : null,
    choose: (selection: Selection) => {
      if (scope === null || client.isMutating(filter) > 0) return;
      save.mutate({ ...scope, selection });
    },
  };
}
