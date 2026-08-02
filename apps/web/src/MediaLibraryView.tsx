import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Image as ImageIcon, Pencil, Search, Trash2 } from "lucide-react";
import { api } from "./api";
import { EmptyState } from "./EmptyState";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";
import type { ArtifactLibraryItem } from "./types";

export function MediaLibraryView({
  onEditImage,
}: {
  onEditImage?: (artifact: ArtifactLibraryItem) => void;
}) {
  const client = useQueryClient();
  const [kind, setKind] = useState("");
  const [search, setSearch] = useState("");
  const artifacts = useQuery({
    queryKey: ["artifacts", kind, search],
    queryFn: () => api.artifacts(kind, search),
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
      </div>
      {cleanup.data && <div className="callout success" role="status">{cleanup.data.removed_count > 0 && <>Removed {cleanup.data.removed_count} artifact{cleanup.data.removed_count === 1 ? "" : "s"} and reclaimed {formatBytes(cleanup.data.reclaimed_bytes)}. </>}{cleanup.data.marked_count > 0 ? `${cleanup.data.marked_count} newly unreferenced artifact${cleanup.data.marked_count === 1 ? "" : "s"} entered the ${storage.data?.retention_days ?? 30}-day recovery window.` : cleanup.data.retention_pending_count > 0 ? `${cleanup.data.retention_pending_count} artifact${cleanup.data.retention_pending_count === 1 ? "" : "s"} ${cleanup.data.retention_pending_count === 1 ? "remains" : "remain"} in the recovery window; none are eligible yet.` : cleanup.data.removed_count === 0 ? "Nothing needed cleanup." : null}</div>}
      {(cleanup.error || deleteArtifact.error) && <ErrorCallout message={cleanup.error?.message || deleteArtifact.error?.message} />}
      {artifacts.data?.length ? <div className="media-grid">{artifacts.data.map((artifact) => {
        const source = `/api/artifacts/${encodeURIComponent(artifact.id)}/content`;
        const proxyId = typeof artifact.metadata_json.browser_proxy_artifact_id === "string" ? artifact.metadata_json.browser_proxy_artifact_id : null;
        const playbackSource = proxyId ? `/api/artifacts/${encodeURIComponent(proxyId)}/content` : source;
        const posterId = typeof artifact.metadata_json.poster_artifact_id === "string" ? artifact.metadata_json.poster_artifact_id : null;
        return <article className="gallery-card" key={artifact.id}>
          {/* Generated media has no caption track to point at, and an empty one would claim an affordance that is not there. */}
          {/* eslint-disable-next-line jsx-a11y-x/media-has-caption */}
          {artifact.kind === "image" ? <img src={source} alt={artifact.original_name ?? "Generated image"} loading="lazy" /> : <video src={playbackSource} poster={posterId ? `/api/artifacts/${encodeURIComponent(posterId)}/content` : undefined} controls preload="metadata" />}
          <div><strong>{artifact.original_name ?? artifact.kind}</strong><small>{formatBytes(artifact.size_bytes)} · {artifact.reference_count} reference{artifact.reference_count === 1 ? "" : "s"}</small><span><a href={source} download>Download</a><code>{artifact.sha256.slice(0, 12)}</code>{artifact.kind === "image" && onEditImage && <button className="icon-button" aria-label={`Edit ${artifact.original_name ?? artifact.kind}`} title="Edit" onClick={() => onEditImage(artifact)}><Pencil size={14} /></button>}<button className="icon-button danger" aria-label={`Delete ${artifact.original_name ?? artifact.kind}`} disabled={deleteArtifact.isPending && deleteArtifact.variables === artifact.id} onClick={() => { const references = artifact.reference_count ? ` and remove ${artifact.reference_count} appearance${artifact.reference_count === 1 ? "" : "s"} from chats` : ""; if (window.confirm(`Permanently delete ${artifact.original_name ?? artifact.kind}${references}?`)) deleteArtifact.mutate(artifact.id); }}><Trash2 size={14} /></button></span></div>
        </article>;
      })}</div> : <EmptyState icon={<ImageIcon />} title="No generated media" body="Generated images and videos appear here." />}
    </div>
  );
}
