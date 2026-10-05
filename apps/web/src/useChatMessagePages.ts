import { useInfiniteQuery, useQueryClient, type InfiniteData } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { api } from "./api";
import type { ChatMessageWindow, Message } from "./types";

export type ChatMessagePageCursor = { before?: string; limit: number };
type ChatMessagePages = InfiniteData<ChatMessageWindow, ChatMessagePageCursor>;

function messagesFromPages(data: ChatMessagePages): Message[] {
  const messages = new Map<string, Message>();
  for (const page of [...data.pages].reverse()) {
    for (const message of page.messages) messages.set(message.id, message);
  }
  return [...messages.values()];
}

/** Messages in conversation order, cut into the newest-first pages the transcript reads. */
export function chatMessageWindowPages(chatId: string, ordered: Message[], older: boolean): ChatMessagePages {
  const pages: ChatMessageWindow[] = [];
  for (let end = ordered.length; end > 0; end -= 40) {
    const start = Math.max(0, end - 40);
    pages.push({ chat_id: chatId, messages: ordered.slice(start, end),
      has_older: start > 0 || older, has_newer: end < ordered.length });
  }
  return {
    pages, pageParams: pages.map((page, index) => ({
      before: index ? pages[index - 1].messages[0].id : undefined,
      limit: index ? page.messages.length : 40,
    })),
  };
}

/** The pages already shown, with a later head's newest page added, or null when it does not continue them.
 *
 * It continues them when its page holds the newest message they show: then
 * everything they hold comes before the new messages. A different branch, or
 * more new messages than one page, starts again from the new head instead.
 */
function carriedPages(chatId: string, shown: ChatMessagePages, latest: ChatMessagePages): ChatMessagePages | null {
  const newest = shown.pages[0]?.messages.at(-1)?.id;
  const page = latest.pages[0];
  if (!newest || !page || !page.messages.some((message) => message.id === newest)) return null;
  const messages = new Map(messagesFromPages(shown).map((message) => [message.id, message]));
  for (const message of page.messages) messages.set(message.id, message);
  return chatMessageWindowPages(chatId, [...messages.values()], shown.pages.at(-1)?.has_older ?? false);
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
      const cached = client.getQueryData<ChatMessagePages>(key);
      // A send may leave a short final page. Keep its width when refreshing.
      const retained = cached?.pages[pages.length]?.messages.length;
      return { before, limit: Math.max(1, Math.min(40, retained ?? 40)) };
    },
    // The head moves whenever a message is added, here or in another window.
    // Until the new head's first page arrives the same chat keeps showing what
    // it showed, so the conversation never empties under the person reading it.
    placeholderData: (previous, previousQuery) => previousQuery?.queryKey[1] === chatId ? previous : undefined,
    select: messagesFromPages,
  });
  const shown = useRef<{ chatId: typeof chatId; headId: string | null }>({ chatId, headId });
  // Older messages someone asked for, until they arrive. A refresh, such as
  // the one a message from another window brings, cancels a read in flight;
  // it is asked for again once nothing else is being read.
  const olderWanted = useRef<{ chatId: typeof chatId; oldest?: string; asked: number } | null>(null);
  const { data, isPlaceholderData, dataUpdatedAt, fetchNextPage, isFetching, hasNextPage, isFetchNextPageError } = query;
  useEffect(() => {
    const before = shown.current;
    if (before.chatId === chatId && before.headId !== headId) {
      const latest = client.getQueryData<ChatMessagePages>(["chat", chatId, "messages", headId]);
      // Wait for the new head's own first page; the placeholder is the old one.
      if (isPlaceholderData || !latest) return;
      shown.current = { chatId, headId };
      const previous = client.getQueryData<ChatMessagePages>(["chat", chatId, "messages", before.headId]);
      // A local send has already carried the pages over; then nothing is added.
      const carried = chatId && previous && latest.pages.length === 1 && carriedPages(chatId, previous, latest);
      if (carried) {
        client.setQueryData<ChatMessagePages>(["chat", chatId, "messages", headId], carried);
        return;
      }
      if (latest.pages.length === 1) olderWanted.current = null;
    } else shown.current = { chatId, headId };
    const wanted = olderWanted.current;
    if (!wanted) return;
    const arrived = data?.some((message, index) => index > 0 && message.id === wanted.oldest);
    // Asking a few times covers refreshes; an answer with nothing older ends it.
    if (wanted.chatId !== chatId || arrived || !hasNextPage || isFetchNextPageError || wanted.asked >= 3) {
      olderWanted.current = null;
    } else if (!isFetching && !isPlaceholderData) {
      wanted.asked += 1;
      void fetchNextPage({ cancelRefetch: false });
    }
  }, [client, chatId, headId, data, isPlaceholderData, dataUpdatedAt, fetchNextPage, isFetching, hasNextPage,
    isFetchNextPageError]);
  return { ...query, loadOlder: () => {
    olderWanted.current = { chatId, oldest: data?.[0]?.id, asked: 0 };
    return query.fetchNextPage({ cancelRefetch: false });
  } };
}
