import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { activeWorkflowCapability, workflowChoiceKind } from "./activeWorkflowCapability";
import { api } from "./api";
import { orderFamilies, servesCapability } from "./workflowFamilies";
import type {
  ChatWorkflowSelectionInput,
  RoutingMode,
  WorkflowFamily,
  WorkflowSelection,
} from "./types";
import type {
  ComposerWorkflowCapability,
  WorkflowChoiceKind,
} from "./activeWorkflowCapability";

type SelectionMutation = {
  chatId: string;
  capability: ComposerWorkflowCapability;
  selection: ChatWorkflowSelectionInput;
};

export type ActiveChatWorkflowSelectionState =
  | { kind: "unresolved"; capability: null }
  | {
      kind: "loading";
      capability: ComposerWorkflowCapability;
    }
  | {
      kind: "read-error";
      capability: ComposerWorkflowCapability;
      error: Error;
      retry: () => void;
    }
  | {
      kind: "ready";
      capability: ComposerWorkflowCapability;
      choiceKind: WorkflowChoiceKind;
      current: WorkflowSelection | undefined;
      currentFamilyId: string | null;
      families: WorkflowFamily[];
      selectedFamilyMissing: boolean;
      saving: boolean;
      saveError: Error | null;
      choose: (selection: ChatWorkflowSelectionInput) => void;
    };

export function useActiveChatWorkflowSelection(
  chatId: string,
  routingMode: RoutingMode,
): ActiveChatWorkflowSelectionState {
  const client = useQueryClient();
  const capability = activeWorkflowCapability(routingMode);
  const families = useQuery({
    queryKey: ["workflow-families", capability],
    queryFn: () => api.workflowFamilies(capability!),
    enabled: capability !== null,
  });
  const selections = useQuery({
    queryKey: ["chat", chatId, "workflow-selections"],
    queryFn: () => api.chatWorkflowSelections(chatId),
    enabled: capability !== null,
  });
  const choose = useMutation({
    mutationFn: ({ chatId: selectedChatId, capability: selectedCapability, selection }:
      SelectionMutation) => api.setChatWorkflowSelection(
        selectedChatId,
        selectedCapability,
        selection,
      ),
    onSuccess: (_saved, variables) => client.invalidateQueries({
      queryKey: ["chat", variables.chatId, "workflow-selections"],
    }),
  });

  if (capability === null) return { kind: "unresolved", capability: null };
  if (families.isLoading || selections.isLoading) {
    return { kind: "loading", capability };
  }
  const readError = (families.error ?? selections.error) as Error | null;
  if (readError) {
    return {
      kind: "read-error",
      capability,
      error: readError,
      retry: () => {
        void families.refetch();
        void selections.refetch();
      },
    };
  }

  const current = selections.data?.find(
    (selection) => selection.selector_capability === capability,
  );
  const available = (families.data ?? []).filter(
    (family) => servesCapability(family, capability),
  );
  const ordered = orderFamilies(available, capability);
  const currentFamilyId = current?.mode === "family"
    ? current.workflow_family_id
    : null;
  const mutationMatchesCurrent = choose.variables?.chatId === chatId
    && choose.variables.capability === capability;

  return {
    kind: "ready",
    capability,
    choiceKind: workflowChoiceKind(current),
    current,
    currentFamilyId,
    families: ordered,
    selectedFamilyMissing: Boolean(
      currentFamilyId && !ordered.some((family) => family.id === currentFamilyId),
    ),
    saving: mutationMatchesCurrent && choose.isPending,
    saveError: mutationMatchesCurrent ? choose.error as Error | null : null,
    choose: (selection) => choose.mutate({ chatId, capability, selection }),
  };
}
