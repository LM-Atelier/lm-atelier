import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import { api } from "./api";
import { EmptyState } from "./EmptyState";
import { ErrorCallout } from "./ErrorCallout";
import type { ReferenceSimilarAsset, ReferenceSubject } from "./types";

/** What one image is for. Closed, because a preparation recipe decides what to
 *  do with an image from its purpose - one nobody implements contributes
 *  nothing, silently. */
const PURPOSES = [
  "identity",
  "appearance",
  "clothing",
  "pose",
  "style",
  "environment",
  "detail",
  "product_view",
  "other",
] as const;

export function ReferenceDetail({
  subject,
  onBack,
}: {
  subject: ReferenceSubject;
  onBack: () => void;
}) {
  const client = useQueryClient();
  const [purpose, setPurpose] = useState<string>("identity");
  const [artifactId, setArtifactId] = useState("");
  const [error, setError] = useState<string | null>(null);
  // Held after an attach rather than shown transiently: the whole point is that
  // the person who just added the image gets to decide what to do about it.
  const [similar, setSimilar] = useState<ReferenceSimilarAsset[]>([]);

  const assets = useQuery({
    queryKey: ["reference-assets", subject.id],
    queryFn: () => api.referenceAssets(subject.id),
  });

  const refresh = () => client.invalidateQueries({ queryKey: ["reference-assets", subject.id] });
  const fail = (reason: unknown) =>
    setError(reason instanceof Error ? reason.message : "That did not work");

  const attach = useMutation({
    mutationFn: () => api.attachReferenceAsset(subject.id, { artifact_id: artifactId, purpose }),
    onSuccess: (result) => {
      setArtifactId("");
      setError(null);
      setSimilar(result.similar);
      void refresh();
    },
    onError: fail,
  });

  const detach = useMutation({
    mutationFn: (assetId: string) => api.detachReferenceAsset(subject.id, assetId),
    onSuccess: () => {
      setSimilar([]);
      void refresh();
    },
    onError: fail,
  });

  const items = assets.data ?? [];

  return (
    <section className="page-view reference-detail" aria-labelledby="reference-detail-heading">
      <header className="page-header">
        <div>
          <h1 id="reference-detail-heading">{subject.name}</h1>
          <p className="muted">
            Written as <code>@{subject.mention_slug}</code> in a chat. {items.length} image
            {items.length === 1 ? "" : "s"}.
          </p>
        </div>
        <div className="row-actions">
          <button className="secondary" onClick={onBack}>
            Back to references
          </button>
        </div>
      </header>

      {error ? (
        <ErrorCallout
          message={error}
          action={
            <button className="secondary compact-button" onClick={() => setError(null)}>
              Dismiss
            </button>
          }
        />
      ) : null}

      {similar.length > 0 ? (
        // Advice, not a refusal. The image was added; this only says the set may
        // now lean toward one look, which the person adding it is best placed
        // to judge.
        <ErrorCallout
          message={
            `That image closely resembles ${similar.length} already here. It was added anyway - ` +
            "remove it if the set is now weighted toward one look."
          }
          action={
            <button className="secondary compact-button" onClick={() => setSimilar([])}>
              Understood
            </button>
          }
        />
      ) : null}

      <div className="row-actions">
        <label>
          Image id
          <input
            value={artifactId}
            aria-label="Artifact id to attach"
            onChange={(event) => setArtifactId(event.target.value)}
          />
        </label>
        <label>
          Purpose
          <select value={purpose} onChange={(event) => setPurpose(event.target.value)}>
            {PURPOSES.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <button
          className="primary"
          disabled={!artifactId.trim() || attach.isPending}
          onClick={() => attach.mutate()}
        >
          Add image
        </button>
      </div>

      {items.length === 0 && !assets.isLoading ? (
        <EmptyState
          icon={<Trash2 />}
          title="No images yet"
          body="Add a few from the media library so generations have something to work from."
        />
      ) : null}

      <ul className="reference-asset-grid">
        {items.map((asset) => (
          <li key={asset.id}>
            <img
              src={`/api/artifacts/${encodeURIComponent(asset.artifact_id)}/content`}
              alt={asset.caption ?? `${subject.name}, ${asset.purpose}`}
              loading="lazy"
            />
            <div className="detail-title">
              <span className="badge">{asset.purpose}</span>
              {/* Unchecked is not a synonym for usable: an image nobody has
                  looked at must not let the set claim a reviewed set's
                  fidelity. */}
              <span className="badge">{asset.validation_state}</span>
            </div>
            <button
              className="secondary compact-button danger"
              aria-label={`Remove image ${asset.sort_order + 1}`}
              onClick={() => detach.mutate(asset.id)}
            >
              <Trash2 />
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
