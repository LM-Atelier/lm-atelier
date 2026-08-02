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

export type EditLineageStep = {
  /** The image entering this step, as an artifact id. */
  artifactId: string;
  /** The user's instruction for this step; empty when the turn had no text. */
  instruction: string;
  /** The user message that carried the step, for provenance. */
  messageId: string;
};

/** The chain of edits behind a result, oldest step first.
 *
 * Each step is a user turn that carried an image as its input reference; the
 * walk continues while that image was itself produced by an earlier turn in
 * this transcript. An uploaded original ends the chain, as does an image
 * whose producer is not in this chat (a fork carries results, not history).
 */
export function editLineageForResult(messages: Message[], resultIndex: number): EditLineageStep[] {
  const steps: EditLineageStep[] = [];
  let index = resultIndex;
  while (index >= 0 && steps.length < 100) {
    const userIndex = latestUserIndexBefore(messages, index);
    if (userIndex < 0) break;
    const user = messages[userIndex];
    const reference = [...user.parts]
      .reverse()
      .find((part) =>
        part.type === "image"
        && Boolean(part.artifact_id)
        && part.metadata_json.input_reference === true,
      );
    if (!reference?.artifact_id) break;
    const instruction = user.parts.find((part) => part.type === "text" && part.text)?.text ?? "";
    steps.unshift({
      artifactId: reference.artifact_id,
      instruction,
      messageId: user.id,
    });
    index = producerIndexOf(messages, reference.artifact_id, userIndex);
  }
  return steps;
}

function latestUserIndexBefore(messages: Message[], resultIndex: number): number {
  for (let index = resultIndex; index >= 0; index -= 1) {
    if (messages[index].role === "user") return index;
  }
  return -1;
}

function producerIndexOf(messages: Message[], artifactId: string, before: number): number {
  for (let index = before - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "assistant") continue;
    const produced = message.parts.some((part) =>
      part.type === "image"
      && part.artifact_id === artifactId
      && part.metadata_json.input_reference !== true
      && part.metadata_json.preview !== true,
    );
    if (produced) return index;
  }
  return -1;
}

/** The edit source shown alongside a generated image: the newest image the
 * user's own turn carried as an input reference. Absent for pure
 * text-to-image turns, so the compare affordance only appears where a
 * comparison exists. */
export function editSourceUrlForResult(messages: Message[], resultIndex: number): string | null {
  for (let index = resultIndex; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "user") continue;
    const reference = [...message.parts]
      .reverse()
      .find((part) =>
        part.type === "image"
        && Boolean(part.artifact_id)
        && part.metadata_json.input_reference === true,
      );
    return reference ? artifactSource(reference.artifact_id) : null;
  }
  return null;
}
