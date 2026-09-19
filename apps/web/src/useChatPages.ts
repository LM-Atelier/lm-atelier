import { useInfiniteQuery, type InfiniteData, type QueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { Chat } from "./types";

const PAGE_SIZE = 50;
const CHAT_PAGES_KEY = ["chats", "pages"] as const;
type ChatPage = { items: Chat[]; nextOffset: number | null };
type ChatPages = InfiniteData<ChatPage, number>;

function flattenPages(data: ChatPages): Chat[] {
  const seen = new Set<string>();
  return data.pages.flatMap((page) => page.items).filter((chat) => {
    if (seen.has(chat.id)) return false;
    seen.add(chat.id);
    return true;
  });
}

export function useChatPages(search = "", includeArchived = false) {
  const query = search.trim();
  return useInfiniteQuery({
    queryKey: [...CHAT_PAGES_KEY, query, includeArchived],
    initialPageParam: 0,
    placeholderData: (previous) => previous,
    queryFn: async ({ pageParam, signal }): Promise<ChatPage> => {
      const items = await api.chats(null, includeArchived, query, {
        limit: PAGE_SIZE, offset: pageParam, searchProjects: true, signal,
      });
      return { items, nextOffset: items.length === PAGE_SIZE ? pageParam + items.length : null };
    },
    getNextPageParam: (lastPage) => lastPage.nextOffset,
    select: flattenPages,
  });
}

export function snapshotChatPages(client: QueryClient) {
  return client.getQueriesData<ChatPages>({ queryKey: CHAT_PAGES_KEY });
}

export function restoreChatPages(client: QueryClient, snapshot: ReturnType<typeof snapshotChatPages>) {
  for (const [key, data] of snapshot) client.setQueryData(key, data);
}

/** Update loaded rows while preserving the server offsets for each page. */
export function changeChatPages(client: QueryClient, change: (chat: Chat) => Chat | null) {
  client.setQueriesData<ChatPages>({ queryKey: CHAT_PAGES_KEY }, (current) => current && ({
    ...current,
    pages: current.pages.map((page) => ({
      ...page,
      items: page.items.flatMap((chat) => {
        const updated = change(chat);
        return updated ? [updated] : [];
      }),
    })),
  }));
}
