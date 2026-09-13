import { useInfiniteQuery, useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import type { PackageReviewState } from "./useWorkflowPackageImport";
import { WorkflowPackageReview } from "./WorkflowPackageReview";
import type { CatalogModel } from "./types";

/** Finding a workflow you do not have yet.
 *
 * The Model Library already has this rhythm - search, page, install - and this
 * follows it rather than inventing a second one. Paging is by cursor because
 * that is what the source gives back; there is no page count and no total,
 * because the provider does not report one and guessing would be a number
 * nobody could act on.
 *
 * WHAT A WORKFLOW CARD DELIBERATELY DOES NOT SHOW. The shared catalog item type
 * was built for models and carries thirty fields, most of which a workflow
 * producer never sets - quantization, parameter counts, required runtime. A
 * shared card would render blanks and read as missing information rather than
 * as information that does not apply. So this renders only the fields a
 * workflow actually has.
 *
 * ADDING ONE IS THE SAME REVIEW AS IMPORTING A FILE. A published workflow is a
 * ComfyUI export somebody else made, so it goes through the package review a
 * downloaded export would: what it needs, preparing it, then importing it.
 * Nothing is added until that review says so.
 */

function WorkflowCard({
  item,
  fetching,
  busy,
  onReview,
}: {
  item: CatalogModel;
  fetching: boolean;
  busy: boolean;
  onReview: () => void;
}) {
  return (
    <article className="workflow-discover-card">
      <header>
        <h3>{item.name}</h3>
        {item.author && <p className="muted">by {item.author}</p>}
      </header>
      <dl className="workflow-discover-facts">
        {item.downloads !== null && item.downloads !== undefined && (
          <div>
            <dt>Downloads</dt>
            <dd>{item.downloads.toLocaleString()}</dd>
          </div>
        )}
        {item.likes !== null && item.likes !== undefined && (
          <div>
            <dt>Likes</dt>
            <dd>{item.likes.toLocaleString()}</dd>
          </div>
        )}
      </dl>
      {item.tags.length > 0 && (
        <ul className="workflow-discover-tags">
          {item.tags.slice(0, 6).map((tag) => (
            <li key={tag}>{tag}</li>
          ))}
        </ul>
      )}
      <button
        type="button"
        className="secondary"
        aria-disabled={busy}
        onClick={() => {
          // aria-disabled rather than disabled: disabling the focused button
          // would drop focus to the page while the graph is being fetched.
          if (!busy) onReview();
        }}
      >
        {fetching ? "Fetching…" : "Review and add"}
      </button>
    </article>
  );
}

export function WorkflowDiscover({ onImported }: { onImported?: () => void }) {
  const [typed, setTyped] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [sort, setSort] = useState("trending");

  const catalog = useInfiniteQuery({
    queryKey: ["workflow-catalog", submitted, sort],
    queryFn: ({ pageParam }) => api.workflowCatalog(submitted, sort, pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
  });

  const items = catalog.data?.pages.flatMap((page) => page.items) ?? [];
  // Any stale page makes the whole list stale: a person reading it cannot tell
  // which rows came from which fetch, so the weaker claim is the honest one.
  const stale = catalog.data?.pages.some((page) => page.stale) ?? false;

  const [review, setReview] = useState<PackageReviewState | null>(null);
  const fetchReview = useMutation({
    mutationFn: async (item: CatalogModel): Promise<PackageReviewState> => {
      const graph = await api.workflowCatalogGraph(item.remote_id, item.provider);
      return {
        analysis: await api.analyzeWorkflowPackage(graph.ui_graph),
        // The published name, not the provider's file name: the review offers
        // it as the workflow's name, and "workflow.json" names nothing.
        fileName: item.name,
        uiGraph: graph.ui_graph,
      };
    },
    onSuccess: setReview,
  });

  return (
    <section className="workflow-discover" aria-label="Discover workflows">
      <form
        className="workflow-discover-search"
        onSubmit={(event) => {
          event.preventDefault();
          setSubmitted(typed.trim());
        }}
      >
        <label>
          <span>Search workflows</span>
          <input
            value={typed}
            onChange={(event) => setTyped(event.target.value)}
            placeholder="What do you want to make?"
          />
        </label>
        <label>
          <span>Sort</span>
          <select value={sort} onChange={(event) => setSort(event.target.value)}>
            <option value="trending">Trending</option>
            <option value="downloads">Most downloaded</option>
            <option value="newest">Newest</option>
          </select>
        </label>
        <button className="primary" type="submit">
          Search
        </button>
      </form>

      {stale && (
        <p className="callout" role="status">
          These results are the last ones we were able to fetch, so they may be out of date.
        </p>
      )}

      {catalog.error && <ErrorCallout message={catalog.error.message} />}

      {fetchReview.error && <ErrorCallout message={fetchReview.error.message} />}

      {catalog.isPending && <p className="muted">Looking…</p>}

      {/* An empty result and a search not yet run are different states, and
          saying "no workflows" before anyone has searched would be a claim
          about the library rather than about the search. */}
      {!catalog.isPending && !catalog.error && items.length === 0 && (
        <p className="muted">
          {submitted
            ? `No workflows match “${submitted}”.`
            : "Search to find workflows you can add."}
        </p>
      )}

      <div className="workflow-discover-results">
        {items.map((item) => (
          <WorkflowCard
            key={`${item.provider}:${item.remote_id}`}
            item={item}
            fetching={fetchReview.isPending && fetchReview.variables === item}
            busy={fetchReview.isPending}
            onReview={() => fetchReview.mutate(item)}
          />
        ))}
      </div>

      {catalog.hasNextPage && (
        <button
          className="secondary"
          onClick={() => void catalog.fetchNextPage()}
          disabled={catalog.isFetchingNextPage}
        >
          {catalog.isFetchingNextPage ? "Loading…" : "Show more"}
        </button>
      )}

      {review && (
        <WorkflowPackageReview
          analysis={review.analysis}
          fileName={review.fileName}
          uiGraph={review.uiGraph}
          onImported={() => {
            setReview(null);
            onImported?.();
          }}
          onClose={() => setReview(null)}
        />
      )}
    </section>
  );
}
