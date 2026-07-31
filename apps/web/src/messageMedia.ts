import type { Artifact, Message, MessagePart } from "./types";

export type MediaOrigin = "uploaded" | "generated" | "edited";

export function artifactSource(artifactId: string | null): string | null {
  return artifactId ? `/api/artifacts/${encodeURIComponent(artifactId)}/content` : null;
}

export function artifactOrigin(artifact?: Artifact | null): MediaOrigin | null {
  return artifact?.metadata_json.uploaded === true ? "uploaded" : null;
}

export function mediaOriginForPart(
  part: MessagePart,
  operation?: string,
  fallback?: MediaOrigin | null,
): MediaOrigin | null {
  if (!part.metadata_json.input_reference) {
    if (operation === "image_to_image") return "edited";
    if (fallback) return fallback;
  }
  return artifactOrigin(part.artifact) ?? fallback ?? null;
}

export function mediaOriginLabel(origin: MediaOrigin | null, kind: "image" | "video"): string {
  if (!origin) return `Attached ${kind}`;
  return `${origin[0].toUpperCase()}${origin.slice(1)} ${kind}`;
}

export function messagePartsForTranscript(
  message: Message,
  hiddenInputArtifactIds?: ReadonlySet<string>,
): MessagePart[] {
  return message.parts.filter((part) => (
    part.type !== "generation_metadata"
    && !(
      part.metadata_json.input_reference === true
      && part.artifact_id
      && hiddenInputArtifactIds?.has(part.artifact_id)
    )
  ));
}

export function priorVisibleMediaByMessage(
  messages: Message[],
): Map<string, ReadonlySet<string>> {
  const priorByMessage = new Map<string, ReadonlySet<string>>();
  const visibleArtifactIds = new Set<string>();
  for (const message of messages) {
    priorByMessage.set(message.id, new Set(visibleArtifactIds));
    for (const part of message.parts) {
      if (
        part.artifact_id
        && (part.type === "image" || part.type === "video")
        && part.metadata_json.preview !== true
      ) {
        visibleArtifactIds.add(part.artifact_id);
      }
    }
  }
  return priorByMessage;
}
