import { useState } from "react";
import { useInfiniteQuery, useMutation } from "@tanstack/react-query";
import { Image as ImageIcon, Pencil, RefreshCw, Search, Star, Trash2 } from "lucide-react";
import { api } from "./api";
import { ArtifactGenerationDetails } from "./ArtifactGenerationDetails";
import {
  ARTIFACT_LIBRARY_PAGE_ERROR,
  flattenArtifactLibraryPages,
  type ArtifactLibraryFilters,
  type ArtifactLibraryKind,
} from "./artifactLibraryPage";
import { clockOptions, useClockChoice } from "./clockPreference";
import { EmptyState } from "./EmptyState";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";
import { MediaLibraryRecovery } from "./MediaLibraryRecovery";
import { PictureFileSettings } from "./PictureFileSettings";
import { useSensitiveMediaChoice } from "./sensitiveMedia";
import { ShieldedMedia } from "./ShieldedMedia";
import { useMediaLibraryRecovery } from "./useMediaLibraryRecovery";
import { useMediaOrganization } from "./useMediaOrganization";
import { MediaOrganizationControls } from "./MediaOrganizationControls";
import { MediaOrganizationManager } from "./MediaOrganizationManager";
import { MediaOrganizationReview } from "./MediaOrganizationReview";

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
  onOpenChat,
}: {
  onEditImage?: (artifactId: string) => void;
  onOpenChat?: (chatId: string) => void;
}) {
  const [filters, setFilters] = useState<ArtifactLibraryFilters>({
    kind: "",
    query: "",
    favorite: false,
  });
  const [epoch, setEpoch] = useState(0);
  const clock = useClockChoice();
  // While pictures are covered, a file name could say as much as the picture,
  // so items are named by their place in the grid instead.
  const shielding = useSensitiveMediaChoice() !== "show";
  const [favoriteFailed, setFavoriteFailed] = useState(false);
  const recovery = useMediaLibraryRecovery();
  const organization = useMediaOrganization((command) => {
    setFilters((current) => ({
      ...current,
      collection_id: command.action === "delete-album" && current.collection_id === command.target.id
        ? undefined : current.collection_id,
      tag_id: (command.action === "delete-tag" || command.action === "merge-tags") && current.tag_id === command.target.id
        ? undefined : current.tag_id,
    }));
    setEpoch((current) => current + 1);
  });

  const replaceFilters = (next: ArtifactLibraryFilters) => {
    setFavoriteFailed(false);
    setEpoch((current) => current + 1);
    setFilters(next);
  };
  const refresh = () => {
    setFavoriteFailed(false);
    setEpoch((current) => current + 1);
    organization.refresh();
  };

  const feed = useInfiniteQuery({
    queryKey: ["artifact-library-v1", filters.kind, filters.query, filters.favorite,
      filters.collection_id, filters.tag_id, PAGE_LIMIT, epoch],
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
      entries = flattenArtifactLibraryPages(feed.data.pages, Boolean(filters.collection_id));
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

      <MediaOrganizationControls filters={filters} onFilters={replaceFilters} organization={organization} />
      {unavailable && <ErrorCallout message={LIBRARY_UNAVAILABLE} />}
      {!unavailable && favoriteFailed && <ErrorCallout message="The favorite change could not be confirmed. The library was refreshed." />}
      {!unavailable && feed.isPending && (
        <div className="loading-line" role="status" aria-label="Loading your media" />
      )}
      {!unavailable && !feed.isPending && entries.length > 0 && (
        <>
          <div className="media-grid">
            {entries.map((entry, index) => {
              const source = `/api/artifacts/${encodeURIComponent(entry.artifact_id)}/content`;
              const name = shielding
                ? `${entry.kind === "image" ? "Picture" : "Video"} ${index + 1}`
                : entry.display_name;
              const favoriteLabel = `${entry.favorite ? "Unfavorite" : "Favorite"} ${name}`;
              return (
                <article className="gallery-card" key={entry.id}>
                  <label><input type="checkbox" aria-label={`Select ${name}`}
                    checked={organization.selection.has(entry.id)}
                    onChange={() => organization.choose(entry)} />Select</label>
                  <ShieldedMedia kind={entry.kind === "image" ? "image" : "video"}>
                    {entry.kind === "image" ? (
                      <img src={source} alt={name} loading="lazy" />
                    ) : (
                      // Published videos have no caption track in EntryV1.
                      // eslint-disable-next-line jsx-a11y-x/media-has-caption
                      <video src={source} aria-label={name} controls preload="metadata" />
                    )}
                  </ShieldedMedia>
                  <div>
                    <strong>{name}</strong>
                    <small>{formatBytes(entry.size_bytes)} · Added {new Date(Math.floor(entry.created_at_epoch_micros / 1000)).toLocaleString(undefined, clockOptions(clock))}</small>
                    <ArtifactGenerationDetails key={entry.artifact_id} artifactId={entry.artifact_id} />
                    {entry.kind === "image" && <PictureFileSettings artifactId={entry.artifact_id} pictureName={name} onOpenChat={onOpenChat} />}
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
                          aria-label={`Edit ${name}`}
                          title="Edit"
                          onClick={() => onEditImage(entry.artifact_id)}
                        >
                          <Pencil size={14} />
                        </button>
                      )}
                      <button className="icon-button" aria-label={`Move ${name} to Recently Deleted`}
                        title="Move to Recently Deleted" aria-disabled={recovery.busy}
                        onClick={() => recovery.choose(entry)}><Trash2 size={14} /></button>
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
      <MediaLibraryRecovery recovery={recovery} />
      {organization.manage && <MediaOrganizationManager organization={organization} />}
      <MediaOrganizationReview organization={organization} />
    </div>
  );
}
