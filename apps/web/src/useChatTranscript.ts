import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo } from "react";
import { api } from "./api";
import type { ChatDetail, ChatTranscriptContext, WebSearch } from "./types";
import { useChatMessagePages } from "./useChatMessagePages";

export interface TranscriptReads {
  context?: ChatTranscriptContext;
  loading: boolean;
  error: Error | null;
  retry: () => void;
  hasOlder: boolean;
  loadingOlder: boolean;
  olderError: Error | null;
  loadOlder: () => Promise<unknown>;
  searchError: Error | null;
  retrySearches: () => void;
  hasMorePending: boolean;
  hasLoadedPending: boolean;
  loadingPending: boolean;
  loadPending: () => void;
}

export function useChatTranscript(chatId: string | null) {
  const metadata = useQuery({ queryKey: ["chat", chatId], enabled: Boolean(chatId),
    queryFn: ({ signal }) => api.chatMetadata(chatId!, signal) });
  const readyId = metadata.data?.id;
  const head = metadata.data?.active_head_message_id ?? null;
  const messages = useChatMessagePages(readyId, head);
  const context = useQuery({ queryKey: ["chat", chatId, "context", head], enabled: Boolean(readyId),
    queryFn: ({ signal }) => api.chatContext(readyId!, head, signal),
    // Like the messages, kept for the same chat while a new head's context loads.
    placeholderData: (previous, previousQuery) => previousQuery?.queryKey[1] === chatId ? previous : undefined });
  const oldest = messages.data?.[0]?.id;
  const history = useInfiniteQuery({
    queryKey: ["chat", chatId, "searches", head, oldest], enabled: Boolean(readyId && oldest),
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam, signal }) => api.chatSearches(readyId!, {
      headId: head, oldestMessageId: oldest, before: pageParam, limit: 40, signal,
    }),
    getNextPageParam: (page) => page.next_before ?? undefined,
  });
  const pending = useInfiniteQuery({
    queryKey: ["chat", chatId, "pending-searches"], enabled: Boolean(readyId),
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam, signal }) => api.chatSearches(readyId!, {
      pendingOnly: true, before: pageParam, limit: 40, signal,
    }),
    getNextPageParam: (page) => page.next_before ?? undefined,
  });
  // Only the displayed message range is drained; pending decisions page separately.
  const { hasNextPage, isFetching, isFetchNextPageError, fetchNextPage } = history;
  useEffect(() => {
    if (hasNextPage && !isFetching && !isFetchNextPageError) {
      void fetchNextPage({ cancelRefetch: false });
    }
  }, [hasNextPage, isFetching, isFetchNextPageError, fetchNextPage]);
  const searches = useMemo(() => {
    const result = new Map<string, WebSearch>();
    for (const page of [...(history.data?.pages ?? []), ...(pending.data?.pages ?? [])]) {
      for (const search of page.searches) result.set(search.run_id, search);
    }
    return [...result.values()];
  }, [history.data, pending.data]);
  const data = useMemo<ChatDetail | undefined>(() => metadata.data && messages.data && context.data ? {
    ...metadata.data, messages: messages.data, web_searches: searches,
  } : undefined, [metadata.data, messages.data, context.data, searches]);
  const reads: TranscriptReads = {
    context: context.data,
    loading: Boolean(chatId) && !data && !metadata.error && !messages.error && !context.error,
    error: metadata.error ?? (messages.isFetchNextPageError ? null : messages.error) ?? context.error,
    retry: () => { void metadata.refetch(); if (readyId) { void messages.refetch(); void context.refetch(); } },
    hasOlder: messages.hasNextPage, loadingOlder: messages.isFetchingNextPage,
    olderError: messages.isFetchNextPageError ? messages.error : null,
    loadOlder: messages.loadOlder,
    searchError: history.error ?? pending.error,
    retrySearches: () => { if (oldest) void history.refetch(); void pending.refetch(); },
    hasMorePending: pending.hasNextPage, loadingPending: pending.isFetchingNextPage,
    hasLoadedPending: (pending.data?.pages.length ?? 0) > 1,
    loadPending: () => { void pending.fetchNextPage({ cancelRefetch: false }); },
  };
  return { data, reads };
}
