import { useMutation, type QueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { changeChatPages, restoreChatPages, snapshotChatPages } from "./useChatPages";
import type { Chat, ChatDetail } from "./types";

/** Write a field of a chat, showing the new value before the server agrees.

Any field: its title, the mode it routes in, the presets bound to it, the
settings it remembers. The optimistic part is why this is its own module rather
than three lines: the value has to reach both the open chat and every loaded
page of the chat list, the drafts a person is still typing have to survive the
round trip, and a refusal has to put all of it back. */
export function useChatFieldUpdate({
  client,
  setChatDrafts,
}: {
  client: QueryClient;
  setChatDrafts: (update: (current: Record<string, Partial<Chat>>) => Record<string, Partial<Chat>>) => void;
}) {
  return useMutation({
    mutationFn: ({ id, values }: { id: string; values: Partial<Chat> }) => api.updateChat(id, values),
    onMutate: async ({ id, values }) => {
      await Promise.all([client.cancelQueries({ queryKey: ["chat", id] }), client.cancelQueries({ queryKey: ["chats"] })]);
      const previousChat = client.getQueryData<ChatDetail>(["chat", id]);
      const previousChats = snapshotChatPages(client);
      client.setQueryData<ChatDetail>(["chat", id], (current) => (
        current ? { ...current, ...values } : current
      ));
      changeChatPages(client, (item) => item.id === id ? { ...item, ...values } : item);
      return { previousChat, previousChats };
    },
    onError: (_error, { id }, context) => {
      setChatDrafts((current) => {
        const next = { ...current };
        delete next[id];
        return next;
      });
      if (context?.previousChat) client.setQueryData(["chat", id], context.previousChat);
      if (context?.previousChats) restoreChatPages(client, context.previousChats);
    },
    onSuccess: (updated, { id, values }) => {
      if (updated) {
        client.setQueryData<ChatDetail>(["chat", id], (current) => (
          current ? { ...current, ...updated } : current
        ));
        changeChatPages(client, (item) => item.id === id ? { ...item, ...updated } : item);
        setChatDrafts((current) => {
          const draft = current[id];
          if (!draft) return current;
          const remaining = { ...draft };
          for (const key of Object.keys(values) as (keyof Chat)[]) {
            if (remaining[key] === values[key]) delete remaining[key];
          }
          const next = { ...current };
          if (Object.keys(remaining).length) next[id] = remaining;
          else delete next[id];
          return next;
        });
      }
    },
    onSettled: (updated, error, { id }) => {
      if (updated || error) {
        void client.invalidateQueries({ queryKey: ["chat", id] });
        void client.invalidateQueries({ queryKey: ["chats"] });
      }
    },
  });
}
