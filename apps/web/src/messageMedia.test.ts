import { describe, expect, it } from "vitest";
import { messagePartsForTranscript, priorVisibleMediaByMessage } from "./messageMedia";
import type { Message, MessagePart } from "./types";

const stamp = "2026-07-31T00:00:00Z";

function message(id: string, parentId: string | null, parts: MessagePart[]): Message {
  return {
    id,
    chat_id: "chat-media",
    parent_id: parentId,
    role: id.startsWith("assistant") ? "assistant" : "user",
    status: "complete",
    parts,
    created_at: stamp,
    updated_at: stamp,
  };
}

function imagePart(id: string, artifactId: string, inputReference = false): MessagePart {
  return {
    id,
    position: 0,
    type: "image",
    text: null,
    artifact_id: artifactId,
    metadata_json: inputReference ? { input_reference: true } : {},
  };
}

describe("message media presentation", () => {
  it("hides a reused input only after that artifact was visible", () => {
    const source = message("assistant-source", null, [imagePart("source", "artifact-source")]);
    const edit = message("user-edit", source.id, [
      imagePart("reused", "artifact-source", true),
      imagePart("fresh", "artifact-fresh", true),
    ]);
    const prior = priorVisibleMediaByMessage([source, edit]);

    expect(messagePartsForTranscript(edit, prior.get(edit.id)).map((part) => part.id)).toEqual([
      "fresh",
    ]);
  });

  it("does not treat previews as previously visible sources", () => {
    const preview = imagePart("preview", "artifact-preview");
    preview.metadata_json.preview = true;
    const pending = message("assistant-preview", null, [preview]);
    const edit = message("user-edit", pending.id, [
      imagePart("source", "artifact-preview", true),
    ]);
    const prior = priorVisibleMediaByMessage([pending, edit]);

    expect(messagePartsForTranscript(edit, prior.get(edit.id))).toHaveLength(1);
  });
});
