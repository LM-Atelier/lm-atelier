import { useMutation, useQueryClient } from "@tanstack/react-query";

import { api } from "./api";

/** The per-message actions that change which conversations exist.
 *
 * These actions all mutate chat history and share cache reconciliation.
 */
export function useMessageActions(
  openChat: (id: string) => void,
  showChatView: (view: "chat") => void,
) {
  const client = useQueryClient();
  const deleteExchange = useMutation({
    mutationFn: (messageId: string) => api.deleteExchange(messageId),
    onSuccess: (result) => {
      void client.invalidateQueries({ queryKey: ["chat", result.chat_id] });
      void client.invalidateQueries({ queryKey: ["work-plans", result.chat_id] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
      void client.invalidateQueries({ queryKey: ["artifacts"] });
    },
  });
  const removeItem = useMutation({
    mutationFn: async (messageId: string) => {
      const impact = await api.chatItemRemovalImpact(messageId);
      return api.removeChatItemContent(
        messageId,
        impact.message_revision_id,
        crypto.randomUUID(),
      );
    },
    onSuccess: (result) => {
      void client.invalidateQueries({ queryKey: ["chat", result.chat_id] });
      void client.invalidateQueries({ queryKey: ["artifacts"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const forkThread = useMutation({
    mutationFn: (messageId: string) => api.forkThread(messageId),
    onSuccess: (chat) => {
      void client.invalidateQueries({ queryKey: ["chats"] });
      openChat(chat.id);
      showChatView("chat");
    },
  });
  return { deleteExchange, removeItem, forkThread };
}
