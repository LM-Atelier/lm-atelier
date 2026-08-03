import { Film, Image as ImageIcon, LoaderCircle, Paperclip, Star } from "lucide-react";
import { CompareButton } from "./CompareButton";
import { LineageButton } from "./LineageButton";
import {
  artifactSource,
  mediaOriginLabel,
  type EditLineageStep,
  type MediaOrigin,
} from "./messageMedia";
import type { MessagePart } from "./types";

export function ArtifactPart({
  part,
  origin,
  onEditImage,
  onAnimateImage,
  onReferenceMedia,
  onToggleFavorite,
  compareSourceUrl,
  lineage,
}: {
  part: MessagePart;
  origin: MediaOrigin | null;
  onEditImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onAnimateImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onReferenceMedia?: (part: MessagePart, origin: MediaOrigin) => void;
  onToggleFavorite?: (part: MessagePart) => void;
  compareSourceUrl?: string | null;
  lineage?: EditLineageStep[];
}) {
  const proxyId = typeof part.metadata_json.browser_proxy_artifact_id === "string" ? part.metadata_json.browser_proxy_artifact_id : null;
  const posterId = typeof part.metadata_json.poster_artifact_id === "string"
    ? part.metadata_json.poster_artifact_id
    : null;
  const source = artifactSource(proxyId ?? part.artifact_id);
  const poster = artifactSource(posterId) ?? undefined;
  if (!part.artifact_id) return null;
  const preview = Boolean(part.metadata_json.preview);
  const inputReference = part.metadata_json.input_reference === true;
  if (!source) {
    return <div className="submission-progress"><LoaderCircle size={16} />Loading media</div>;
  }
  if (part.type === "attachment") {
    const name = part.artifact?.original_name || "Attachment";
    return <a className="message-attachment" href={source} download><Paperclip size={14} />{name}</a>;
  }
  const kind = part.type === "video" ? "video" : "image";
  const label = preview ? "Generation preview" : mediaOriginLabel(
    origin,
    kind,
  );
  const callbackOrigin = origin ?? (inputReference ? "uploaded" : "generated");
  if (part.type === "image") {
    return (
      <figure className={`media-card ${preview ? "preview" : ""}`}>
        <img src={source} alt={label} loading="lazy" />
        <figcaption>
          <ImageIcon size={14} /> {label}
          {!preview && onEditImage && <button type="button" onClick={() => onEditImage(part, callbackOrigin)}>Edit</button>}
          {!preview && onAnimateImage && <button type="button" onClick={() => onAnimateImage(part, callbackOrigin)}>Animate</button>}
          {!preview && onReferenceMedia && <button type="button" onClick={() => onReferenceMedia(part, callbackOrigin)}>Reference</button>}
          {!preview && compareSourceUrl && source && <CompareButton before={compareSourceUrl} after={source} />}
          {!preview && lineage && source && <LineageButton steps={lineage} resultUrl={source} />}
          {!preview && onToggleFavorite && (
            <button
              type="button"
              aria-label={part.artifact?.favorite ? "Unfavorite this image" : "Favorite this image"}
              aria-pressed={Boolean(part.artifact?.favorite)}
              onClick={() => onToggleFavorite(part)}
            >
              <Star size={14} fill={part.artifact?.favorite ? "currentColor" : "none"} aria-hidden="true" />
            </button>
          )}
          {!preview && <a href={source} download>Download</a>}
        </figcaption>
      </figure>
    );
  }
  return (
    <figure className="media-card">
      {/* Generated media has no caption track to point at, and an empty one would claim an affordance that is not there. */}
      {/* eslint-disable-next-line jsx-a11y-x/media-has-caption */}
      <video src={source} poster={poster} controls preload="metadata" aria-label={label} />
      <figcaption>
        <Film size={14} /> {label}
        <a href={source} download>Download</a>
      </figcaption>
    </figure>
  );
}
