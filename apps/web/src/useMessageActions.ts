import { useMutation, useQueryClient } from "@tanstack/react-query";

import { api } from "./api";

/** The per-message actions that change which conversations exist.
 *
 * Deleting a turn removes messages, runs, jobs, a possible work plan, and
 * artifact references in one server-side cascade; forking creates a new chat
 * and opens it, because the user is starting a tangent and would otherwise
 * have to hunt for the thread they just made. Both invalidate the same
 * queries, so they live together rather than duplicating that list.
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
  const forkThread = useMutation({
    mutationFn: (messageId: string) => api.forkThread(messageId),
    onSuccess: (chat) => {
      void client.invalidateQueries({ queryKey: ["chats"] });
      openChat(chat.id);
      showChatView("chat");
    },
  });
  return { deleteExchange, forkThread };
}
