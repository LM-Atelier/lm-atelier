import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { ChatDetail, EditedBranch } from "./types";

export function useEditedBranches(chat?: ChatDetail) {
  const client = useQueryClient();
  const [selection, setSelection] = useState<{ chatId: string; branch: EditedBranch } | null>(null);
  const query = useQuery({
    queryKey: ["edited-branches", chat?.id],
    enabled: Boolean(chat),
    queryFn: async ({ signal }) => {
      const branches: EditedBranch[] = [];
      const cursors = new Set<string>();
      let cursor: string | null = null;
      do {
        const page = await api.editedBranches(chat!.id, cursor, signal);
        branches.push(...page.items);
        cursor = page.next_cursor;
        if (cursor !== null && cursors.has(cursor)) {
          throw new Error("Edited versions could not be loaded.");
        }
        if (cursor !== null) cursors.add(cursor);
      } while (cursor !== null);
      return branches;
    },
    refetchInterval: 5000,
  });
  const activation = useMutation({
    mutationFn: ({ branch, chatId, expectedHead }: {
      branch: EditedBranch; chatId: string; expectedHead: string | null;
    }) => api.activateEditedBranch(chatId, branch.plan.id, expectedHead),
    onSuccess: (result) => {
      client.setQueryData<ChatDetail>(["chat", result.chat_id], (current) => current ? {
        ...current, active_head_message_id: result.active_head_message_id,
      } : current);
      setSelection((current) => current?.chatId === result.chat_id ? null : current);
      void client.invalidateQueries({ queryKey: ["chat", result.chat_id] });
      void client.invalidateQueries({ queryKey: ["chats"] });
      void client.invalidateQueries({ queryKey: ["edited-branches", result.chat_id] });
    },
  });
  const preview = selection?.chatId === chat?.id ? (
    query.data?.find((branch) => branch.plan.id === selection?.branch.plan.id)
      ?? selection?.branch ?? null
  ) : null;
  return {
    branches: query.data ?? [],
    failed: query.isError,
    preview,
    view: (branch: EditedBranch) => {
      if (!chat) return;
      setSelection({ chatId: chat.id, branch });
      void client.invalidateQueries({ queryKey: ["chat", chat.id] });
    },
    close: () => setSelection(null),
    continueBranch: (branch: EditedBranch) => {
      if (!chat || !branch.can_continue || activation.isPending) return;
      activation.mutate({ branch, chatId: chat.id, expectedHead: chat.active_head_message_id });
    },
    activating: activation.isPending,
    activatingPlanId: activation.isPending ? activation.variables?.branch.plan.id : null,
    activationFailed: activation.isError,
  };
}
