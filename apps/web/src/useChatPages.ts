import { useInfiniteQuery, type InfiniteData, type QueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { ChatSummary } from "./types";

const PAGE_SIZE = 50;
const CHAT_PAGES_KEY = ["chats", "summaries"] as const;
type ChatPage = { items: ChatSummary[]; nextOffset: number | null };
type ChatPages = InfiniteData<ChatPage, number>;

function flattenPages(data: ChatPages): ChatSummary[] {
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
    refetchInterval: 3_000,
    queryFn: async ({ pageParam, signal }): Promise<ChatPage> => {
      const items = await api.chatSummaries(null, includeArchived, query, {
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
export function changeChatPages(client: QueryClient, change: (chat: ChatSummary) => ChatSummary | null) {
  client.setQueriesData<ChatPages>({ queryKey: CHAT_PAGES_KEY }, (current) => current && ({
    ...current,
    pages: current.pages.map((page) => ({
      ...page,
      items: page.items.flatMap((chat) => {
        const updated = change(chat);
        return updated ? [{
          id: updated.id, project_id: updated.project_id, title: updated.title,
          archived: updated.archived, pinned: updated.pinned,
          created_at: updated.created_at, updated_at: updated.updated_at, activity: updated.activity,
        }] : [];
      }),
    })),
  }));
}
