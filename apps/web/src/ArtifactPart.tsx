import {
  Download,
  Film,
  Image as ImageIcon,
  LoaderCircle,
  Paperclip,
  Pencil,
  Quote,
  Star,
} from "lucide-react";
import { CompareButton } from "./CompareButton";
import { ImageStudioIcon } from "./ImageStudioIcon";
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
  onOpenStudio,
  onAnimateImage,
  onReferenceMedia,
  onToggleFavorite,
  compareSourceUrl,
  lineage,
}: {
  part: MessagePart;
  origin: MediaOrigin | null;
  onEditImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onOpenStudio?: (part: MessagePart) => void;
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
        {/* A picture narrower than the card used to sit between flat bars.
            The bars are now the picture itself, blurred and overscaled, so
            the frame reads as full without cropping anything or pretending
            the image is a shape it is not. Decorative, so it is hidden from
            assistive technology - the real image carries the description. */}
        <div className="media-frame">
          <img className="media-backdrop" src={source} alt="" aria-hidden="true" loading="lazy" />
          <img src={source} alt={label} loading="lazy" />
        </div>
        <figcaption>
          <ImageIcon size={14} /> {label}
          {/* Icons, not labels - but each operation stays distinct: Edit
              attaches and selects image mode, Animate attaches and seeds a
              video turn, Reference attaches without changing the mode. */}
          {!preview && onEditImage && (
            <button
              type="button"
              className="icon-button"
              aria-label="Edit this image"
              title="Edit"
              onClick={() => onEditImage(part, callbackOrigin)}
            >
              <Pencil size={14} aria-hidden="true" />
            </button>
          )}
          {!preview && onOpenStudio && (
            <button
              type="button"
              className="icon-button"
              aria-label="Open this image in the Image Studio"
              title="Open in Image Studio"
              onClick={() => onOpenStudio(part)}
            >
              <ImageStudioIcon size={14} aria-hidden="true" />
            </button>
          )}
          {!preview && onAnimateImage && (
            <button
              type="button"
              className="icon-button"
              aria-label="Animate this image"
              title="Animate"
              onClick={() => onAnimateImage(part, callbackOrigin)}
            >
              <Film size={14} aria-hidden="true" />
            </button>
          )}
          {!preview && onReferenceMedia && (
            <button
              type="button"
              className="icon-button"
              aria-label="Reference this media"
              title="Reference"
              onClick={() => onReferenceMedia(part, callbackOrigin)}
            >
              <Quote size={14} aria-hidden="true" />
            </button>
          )}
          {!preview && compareSourceUrl && source && <CompareButton before={compareSourceUrl} after={source} />}
          {!preview && lineage && source && <LineageButton steps={lineage} resultUrl={source} />}
          {!preview && onToggleFavorite && (
            <button
              type="button"
              className="icon-button"
              aria-label={part.artifact?.favorite ? "Unfavorite this image" : "Favorite this image"}
              aria-pressed={Boolean(part.artifact?.favorite)}
              onClick={() => onToggleFavorite(part)}
            >
              <Star size={14} fill={part.artifact?.favorite ? "currentColor" : "none"} aria-hidden="true" />
            </button>
          )}
          {!preview && (
            <a className="icon-button" href={source} download aria-label="Download this image" title="Download">
              <Download size={14} aria-hidden="true" />
            </a>
          )}
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
