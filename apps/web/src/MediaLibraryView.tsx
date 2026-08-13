import { useState } from "react";
import { useInfiniteQuery, useMutation } from "@tanstack/react-query";
import { Image as ImageIcon, Pencil, RefreshCw, Search, Star } from "lucide-react";
import { api } from "./api";
import {
  ARTIFACT_LIBRARY_PAGE_ERROR,
  flattenArtifactLibraryPages,
  type ArtifactLibraryFilters,
  type ArtifactLibraryKind,
} from "./artifactLibraryPage";
import { EmptyState } from "./EmptyState";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";

const PAGE_LIMIT = 20;
const LIBRARY_UNAVAILABLE = "The Media Library could not be loaded safely. Refresh and try again.";
const MAX_QUERY_CODE_POINTS = 200;

function boundedQuery(value: string): string | null {
  let result = "";
  let count = 0;
  for (const character of value) {
    const code = character.codePointAt(0);
    if (code === undefined || (character.length === 1 && code >= 0xd800 && code <= 0xdfff)) {
      return null;
    }
    if (count === MAX_QUERY_CODE_POINTS) break;
    result += character;
    count += 1;
  }
  return result;
}

export function MediaLibraryView({
  onEditImage,
}: {
  onEditImage?: (artifactId: string) => void;
}) {
  const [filters, setFilters] = useState<ArtifactLibraryFilters>({
    kind: "",
    query: "",
    favorite: false,
  });
  const [epoch, setEpoch] = useState(0);
  const [favoriteFailed, setFavoriteFailed] = useState(false);

  const replaceFilters = (next: ArtifactLibraryFilters) => {
    setFavoriteFailed(false);
    setEpoch((current) => current + 1);
    setFilters(next);
  };
  const refresh = () => {
    setFavoriteFailed(false);
    setEpoch((current) => current + 1);
  };

  const feed = useInfiniteQuery({
    queryKey: ["artifact-library-v1", filters.kind, filters.query, filters.favorite, PAGE_LIMIT, epoch],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) =>
      api.artifactLibrary(filters, pageParam, PAGE_LIMIT, signal),
    getNextPageParam: (page) => page.next_cursor ?? undefined,
    retry: false,
  });

  const favorite = useMutation({
    mutationFn: ({ artifactId, next }: { artifactId: string; next: boolean }) =>
      api.favoriteArtifact(artifactId, next),
    onMutate: () => setFavoriteFailed(false),
    onError: () => setFavoriteFailed(true),
    onSettled: () => setEpoch((current) => current + 1),
  });

  let entries = [] as ReturnType<typeof flattenArtifactLibraryPages>;
  let chainInvalid = false;
  if (feed.data) {
    try {
      entries = flattenArtifactLibraryPages(feed.data.pages);
    } catch (error) {
      if (!(error instanceof Error) || error.message !== ARTIFACT_LIBRARY_PAGE_ERROR) throw error;
      chainInvalid = true;
    }
  }
  const unavailable = chainInvalid || Boolean(feed.error);
  const busy = feed.isPending || feed.isFetching || favorite.isPending;

  return (
    <div className="page-view media-library">
      <header className="page-header">
        <div>
          <h1>Media library</h1>
          <p>Durable images and videos you have published to your library.</p>
        </div>
        <button className="secondary" onClick={refresh} disabled={busy}>
          <RefreshCw size={14} /> Refresh
        </button>
      </header>
      <div className="media-toolbar">
        <div className="workspace-search">
          <Search size={14} />
          <input
            aria-label="Search media"
            placeholder="Search display names"
            value={filters.query}
            maxLength={MAX_QUERY_CODE_POINTS * 2}
            onChange={(event) => {
              const query = boundedQuery(event.currentTarget.value);
              event.currentTarget.value = query ?? filters.query;
              if (query === null || query === filters.query) return;
              replaceFilters({ ...filters, query });
            }}
          />
        </div>
        <select
          aria-label="Media type"
          value={filters.kind}
          onChange={(event) => replaceFilters({
            ...filters,
            kind: event.target.value as "" | ArtifactLibraryKind,
          })}
        >
          <option value="">Images and videos</option>
          <option value="image">Images</option>
          <option value="video">Videos</option>
        </select>
        <select
          aria-label="Favorites filter"
          value={filters.favorite ? "favorites" : "all"}
          onChange={(event) => replaceFilters({
            ...filters,
            favorite: event.target.value === "favorites",
          })}
        >
          <option value="all">All media</option>
          <option value="favorites">Favorites</option>
        </select>
      </div>

      {unavailable && <ErrorCallout message={LIBRARY_UNAVAILABLE} />}
      {!unavailable && favoriteFailed && <ErrorCallout message="The favorite change could not be confirmed. The library was refreshed." />}
      {!unavailable && feed.isPending && (
        <div className="loading-line" role="status" aria-label="Loading your media" />
      )}
      {!unavailable && !feed.isPending && entries.length > 0 && (
        <>
          <div className="media-grid">
            {entries.map((entry) => {
              const source = `/api/artifacts/${encodeURIComponent(entry.artifact_id)}/content`;
              const favoriteLabel = `${entry.favorite ? "Unfavorite" : "Favorite"} ${entry.display_name}`;
              return (
                <article className="gallery-card" key={entry.id}>
                  {entry.kind === "image" ? (
                    <img src={source} alt={entry.display_name} loading="lazy" />
                  ) : (
                    // Published videos have no caption track in EntryV1.
                    // eslint-disable-next-line jsx-a11y-x/media-has-caption
                    <video src={source} aria-label={entry.display_name} controls preload="metadata" />
                  )}
                  <div>
                    <strong>{entry.display_name}</strong>
                    <small>{formatBytes(entry.size_bytes)} · Added {new Date(Math.floor(entry.created_at_epoch_micros / 1000)).toLocaleString()}</small>
                    <span>
                      <button
                        className={`icon-button ${entry.favorite ? "favorite-active" : ""}`}
                        aria-label={favoriteLabel}
                        aria-pressed={entry.favorite}
                        title={entry.favorite ? "Unfavorite" : "Favorite"}
                        disabled={favorite.isPending}
                        onClick={() => favorite.mutate({
                          artifactId: entry.artifact_id,
                          next: !entry.favorite,
                        })}
                      >
                        <Star size={14} fill={entry.favorite ? "currentColor" : "none"} />
                      </button>
                      {entry.kind === "image" && onEditImage && (
                        <button
                          className="icon-button"
                          aria-label={`Edit ${entry.display_name}`}
                          title="Edit"
                          onClick={() => onEditImage(entry.artifact_id)}
                        >
                          <Pencil size={14} />
                        </button>
                      )}
                    </span>
                  </div>
                </article>
              );
            })}
          </div>
          {feed.hasNextPage && (
            <div>
              <p role="status">Showing the newest {entries.length} items. More are available.</p>
              <button
                className="secondary"
                disabled={feed.isFetchingNextPage}
                onClick={() => void feed.fetchNextPage()}
              >
                {feed.isFetchingNextPage ? "Loading…" : "Load more"}
              </button>
            </div>
          )}
        </>
      )}
      {!unavailable && !feed.isPending && entries.length === 0 && (
        <EmptyState
          icon={<ImageIcon />}
          title="No media matches these filters"
          body="Published images and videos appear here."
        />
      )}
    </div>
  );
}
