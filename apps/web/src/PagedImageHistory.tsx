import { useInfiniteQuery } from "@tanstack/react-query";
import { api } from "./api";
import { CompareButton } from "./CompareButton";
import { LineageButton } from "./LineageButton";
import { artifactSource } from "./messageMedia";

export interface ImageHistoryTarget { chatId: string; resultId: string }

export function PagedImageHistory({ target, resultUrl }: { target: ImageHistoryTarget; resultUrl: string }) {
  const history = useInfiniteQuery({
    queryKey: ["chat", target.chatId, "edit-lineage", target.resultId],
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam, signal }) => api.chatEditLineage(target.chatId, target.resultId, {
      before: pageParam, limit: pageParam ? 40 : 2, signal,
    }),
    getNextPageParam: (page) => page.next_before ?? undefined,
  });
  const steps = [...new Map((history.data?.pages.flatMap((page) => page.steps) ?? [])
    .map((step) => [step.message_id, step])).values()];
  const before = artifactSource(steps[0]?.artifact_id ?? null);
  return <>
    {history.isPending && <span role="status">Loading image history…</span>}
    {history.isError && !history.isFetchNextPageError && <span role="alert">Image history could not be loaded.
      <button type="button" onClick={() => { void history.refetch(); }}>Retry image history</button>
    </span>}
    {before && <CompareButton before={before} after={resultUrl} />}
    <LineageButton resultUrl={resultUrl} steps={[...steps].reverse().map((step) => ({
      artifactId: step.artifact_id, messageId: step.message_id, instruction: step.instruction,
    }))} paging={{
      hasOlder: history.hasNextPage, loading: history.isFetching,
      error: history.isFetchNextPageError,
      onOlder: () => { if (!history.isFetching && history.hasNextPage) void history.fetchNextPage({ cancelRefetch: false }); },
    }} />
  </>;
}
