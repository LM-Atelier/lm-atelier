import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Image as ImageIcon, Pencil, Search, Star, Trash2 } from "lucide-react";
import { api } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
import { EmptyState } from "./EmptyState";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";
import { GenerationIdentitySummary } from "./GenerationIdentitySummary";
import type { ArtifactLibraryItem } from "./types";

export function MediaLibraryView({
  onEditImages,
}: {
  onEditImages?: (artifacts: ArtifactLibraryItem[]) => void;
}) {
  const client = useQueryClient();
  const [deleting, setDeleting] = useState<ArtifactLibraryItem | null>(null);
  const [kind, setKind] = useState("");
  const [search, setSearch] = useState("");
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  // A multi-select over image cards; the batch goes to the editing studio
  // together and "Apply to each" runs one turn per image.
  const [selectedIds, setSelectedIds] = useState<ReadonlySet<string>>(new Set());
  const toggleSelected = (artifactId: string) => setSelectedIds((current) => {
    const next = new Set(current);
    if (next.has(artifactId)) next.delete(artifactId);
    else next.add(artifactId);
    return next;
  });
  const selectedArtifacts = (artifacts: ArtifactLibraryItem[] | undefined) =>
    (artifacts ?? []).filter((artifact) => selectedIds.has(artifact.id));
  const favorite = useMutation({
    mutationFn: ({ artifactId, next }: { artifactId: string; next: boolean }) =>
      api.favoriteArtifact(artifactId, next),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["artifacts"] }),
  });
  const artifacts = useQuery({
    queryKey: ["artifacts", kind, search, favoritesOnly],
    queryFn: () => api.artifacts(kind, search, favoritesOnly),
  });
  const storage = useQuery({ queryKey: ["artifact-storage"], queryFn: api.artifactStorage });
  const cleanup = useMutation({
    mutationFn: () => api.cleanupArtifacts(false),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["artifacts"] });
      void client.invalidateQueries({ queryKey: ["artifact-storage"] });
    },
  });
  const deleteArtifact = useMutation({
    mutationFn: api.deleteArtifact,
    onMutate: async (artifactId: string) => {
      await client.cancelQueries({ queryKey: ["artifacts"] });
      const previous = client.getQueriesData<ArtifactLibraryItem[]>({ queryKey: ["artifacts"] });
      for (const [queryKey, items] of previous) {
        client.setQueryData(
          queryKey,
          items?.filter((artifact) => artifact.id !== artifactId),
        );
      }
      return { previous };
    },
    onError: (_error, _artifactId, context) => {
      for (const [queryKey, items] of context?.previous ?? []) {
        client.setQueryData(queryKey, items);
      }
    },
    onSettled: () => {
      void client.invalidateQueries({ queryKey: ["artifacts"] });
      void client.invalidateQueries({ queryKey: ["artifact-storage"] });
      void client.invalidateQueries({ queryKey: ["chat"] });
    },
  });
  return (
    <div className="page-view media-library">
      <header className="page-header">
        <div><h1>Media library</h1></div>
      </header>
      {storage.data && <section className={`artifact-storage-summary ${storage.data.warning ? "warning" : ""}`}>
        <div><strong>{formatBytes(storage.data.total_bytes)}</strong><small>{storage.data.total_count} stored artifacts</small></div>
        <div><strong>{formatBytes(storage.data.referenced_bytes)}</strong><small>{storage.data.referenced_count} referenced</small></div>
        <div><strong>{formatBytes(storage.data.disk_free_bytes)}</strong><small>disk available</small></div>
        <div><strong>{formatBytes(storage.data.eligible_bytes)}</strong><small>{storage.data.eligible_count} eligible for cleanup</small></div>
        <button className="secondary" disabled={(!storage.data.eligible_count && !(storage.data.retention_pending_count ?? 0)) || cleanup.isPending} onClick={() => cleanup.mutate()}>{cleanup.isPending ? "Cleaning…" : "Run cleanup"}</button>
      </section>}
      <div className="media-toolbar">
        <div className="workspace-search"><Search size={14} /><input aria-label="Search media" placeholder="Search filenames or hashes" value={search} onChange={(event) => setSearch(event.target.value)} /></div>
        <select aria-label="Media type" value={kind} onChange={(event) => setKind(event.target.value)}><option value="">Images and videos</option><option value="image">Images</option><option value="video">Videos</option></select>
        <select aria-label="Favorites filter" value={favoritesOnly ? "favorites" : "all"} onChange={(event) => setFavoritesOnly(event.target.value === "favorites")}><option value="all">All media</option><option value="favorites">Favorites</option></select>
      </div>
      {onEditImages && selectedIds.size > 0 && (
        <div className="media-selection-bar" role="toolbar" aria-label="Selected images">
          <span>{selectedIds.size} image{selectedIds.size === 1 ? "" : "s"} selected</span>
          <button
            className="secondary compact-button"
            onClick={() => { onEditImages(selectedArtifacts(artifacts.data)); setSelectedIds(new Set()); }}
          >
            Edit {selectedIds.size === 1 ? "it" : "together"} in the studio
          </button>
          <button className="secondary compact-button" onClick={() => setSelectedIds(new Set())}>
            Clear selection
          </button>
        </div>
      )}
      {cleanup.data && <div className="callout success" role="status">{cleanup.data.removed_count > 0 && <>Removed {cleanup.data.removed_count} artifact{cleanup.data.removed_count === 1 ? "" : "s"} and reclaimed {formatBytes(cleanup.data.reclaimed_bytes)}. </>}{cleanup.data.marked_count > 0 ? `${cleanup.data.marked_count} newly unreferenced artifact${cleanup.data.marked_count === 1 ? "" : "s"} entered the ${storage.data?.retention_days ?? 30}-day recovery window.` : cleanup.data.retention_pending_count > 0 ? `${cleanup.data.retention_pending_count} artifact${cleanup.data.retention_pending_count === 1 ? "" : "s"} ${cleanup.data.retention_pending_count === 1 ? "remains" : "remain"} in the recovery window; none are eligible yet.` : cleanup.data.removed_count === 0 ? "Nothing needed cleanup." : null}</div>}
      {(cleanup.error || deleteArtifact.error || favorite.error) && <ErrorCallout message={(cleanup.error ?? deleteArtifact.error ?? favorite.error)!.message} />}
      {/* "No generated media" while the request is still in flight tells the
          user something untrue about their own library. */}
      {artifacts.isLoading && <div className="loading-line" role="status" aria-label="Loading your media" />}
      {/* And a request that failed says even less true a thing: the library
          is not empty, it is unreachable. Same reasoning as the line above,
          which only ever covered the half of it that was still in flight. */}
      {artifacts.error && <ErrorCallout message={(artifacts.error as Error).message} />}
      {artifacts.isLoading || artifacts.error ? null : artifacts.data?.length ? <div className="media-grid">{artifacts.data.map((artifact) => {
        const source = `/api/artifacts/${encodeURIComponent(artifact.id)}/content`;
        const proxyId = typeof artifact.metadata_json.browser_proxy_artifact_id === "string" ? artifact.metadata_json.browser_proxy_artifact_id : null;
        const playbackSource = proxyId ? `/api/artifacts/${encodeURIComponent(proxyId)}/content` : source;
        const posterId = typeof artifact.metadata_json.poster_artifact_id === "string" ? artifact.metadata_json.poster_artifact_id : null;
        return <article className="gallery-card" key={artifact.id}>
          {/* Generated media has no caption track to point at, and an empty one would claim an affordance that is not there. */}
          {/* eslint-disable-next-line jsx-a11y-x/media-has-caption */}
          {artifact.kind === "image" ? <img src={source} alt={artifact.original_name ?? "Generated image"} loading="lazy" /> : <video src={playbackSource} poster={posterId ? `/api/artifacts/${encodeURIComponent(posterId)}/content` : undefined} controls preload="metadata" />}
          <div><strong>{artifact.original_name ?? artifact.kind}</strong><small>{formatBytes(artifact.size_bytes)} · {artifact.reference_count} reference{artifact.reference_count === 1 ? "" : "s"}</small><GenerationIdentitySummary identity={artifact.generation_identity} /><span><a href={source} download>Download</a><code>{artifact.sha256.slice(0, 12)}</code><button className={`icon-button ${artifact.favorite ? "favorite-active" : ""}`} aria-label={artifact.favorite ? `Unfavorite ${artifact.original_name ?? artifact.kind}` : `Favorite ${artifact.original_name ?? artifact.kind}`} aria-pressed={artifact.favorite} title={artifact.favorite ? "Unfavorite" : "Favorite"} onClick={() => favorite.mutate({ artifactId: artifact.id, next: !artifact.favorite })}><Star size={14} fill={artifact.favorite ? "currentColor" : "none"} /></button>{artifact.kind === "image" && onEditImages && <input type="checkbox" aria-label={`Select ${artifact.original_name ?? artifact.kind}`} checked={selectedIds.has(artifact.id)} onChange={() => toggleSelected(artifact.id)} />}{artifact.kind === "image" && onEditImages && <button className="icon-button" aria-label={`Edit ${artifact.original_name ?? artifact.kind}`} title="Edit" onClick={() => onEditImages([artifact])}><Pencil size={14} /></button>}<button className="icon-button danger" aria-label={`Delete ${artifact.original_name ?? artifact.kind}`} disabled={deleteArtifact.isPending && deleteArtifact.variables === artifact.id} onClick={() => setDeleting(artifact)}><Trash2 size={14} /></button></span></div>
        </article>;
      })}</div> : <EmptyState icon={<ImageIcon />} title="No generated media" body="Generated images and videos appear here." />}
      {deleting && (
        <ConfirmDialog
          title={`Permanently delete ${deleting.original_name ?? deleting.kind}?`}
          question={
            deleting.reference_count
              ? `This also removes ${deleting.reference_count} appearance${deleting.reference_count === 1 ? "" : "s"} from your chats. The file cannot be recovered.`
              : "The file cannot be recovered."
          }
          confirmLabel="Delete permanently"
          onCancel={() => setDeleting(null)}
          onConfirm={() => {
            const chosen = deleting;
            setDeleting(null);
            deleteArtifact.mutate(chosen.id);
          }}
        />
      )}
    </div>
  );
}
