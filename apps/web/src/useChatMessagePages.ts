import { useInfiniteQuery, useQueryClient, type InfiniteData } from "@tanstack/react-query";
import { api } from "./api";
import type { ChatMessageWindow, Message } from "./types";

export type ChatMessagePageCursor = { before?: string; limit: number };

function messagesFromPages(data: InfiniteData<ChatMessageWindow, ChatMessagePageCursor>): Message[] {
  const messages = new Map<string, Message>();
  for (const page of [...data.pages].reverse()) {
    for (const message of page.messages) messages.set(message.id, message);
  }
  return [...messages.values()];
}

export function useChatMessagePages(chatId: string | null | undefined, headId: string | null) {
  const client = useQueryClient();
  const key = ["chat", chatId, "messages", headId];
  const query = useInfiniteQuery({
    queryKey: key,
    enabled: Boolean(chatId),
    initialPageParam: { limit: 40 } as ChatMessagePageCursor,
    queryFn: ({ pageParam, signal }) => api.chatMessages(chatId!, {
      headId, before: pageParam.before, limit: pageParam.limit, signal,
    }),
    getNextPageParam: (page, pages) => {
      const before = page.has_older ? page.messages[0]?.id : undefined;
      if (!before) return undefined;
      const cached = client.getQueryData<InfiniteData<ChatMessageWindow, ChatMessagePageCursor>>(key);
      // A send may leave a short final page. Keep its width when refreshing.
      const retained = cached?.pages[pages.length]?.messages.length;
      return { before, limit: Math.max(1, Math.min(40, retained ?? 40)) };
    },
    select: messagesFromPages,
  });
  return { ...query, loadOlder: () => query.fetchNextPage({ cancelRefetch: false }) };
}
