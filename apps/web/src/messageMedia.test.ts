import { describe, expect, it } from "vitest";
import { editLineageForResult, messagePartsForTranscript, priorVisibleMediaByMessage } from "./messageMedia";
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

describe("edit lineage", () => {
  const textPart = (id: string, text: string): MessagePart => ({
    id,
    position: 0,
    type: "text",
    text,
    artifact_id: null,
    metadata_json: {},
  });
  const chain = [
    message("user-1", null, [
      textPart("t1", "Make it a watercolor"),
      imagePart("ref-1", "artifact-upload", true),
    ]),
    message("assistant-1", "user-1", [imagePart("out-1", "artifact-step-1")]),
    message("user-2", "assistant-1", [
      textPart("t2", "Now add a lighthouse"),
      imagePart("ref-2", "artifact-step-1", true),
    ]),
    message("assistant-2", "user-2", [imagePart("out-2", "artifact-step-2")]),
  ];

  it("walks a two-step chain oldest first with each instruction", () => {
    expect(editLineageForResult(chain, 3)).toEqual([
      { artifactId: "artifact-upload", instruction: "Make it a watercolor", messageId: "user-1" },
      { artifactId: "artifact-step-1", instruction: "Now add a lighthouse", messageId: "user-2" },
    ]);
  });

  it("reports a single step for a first edit", () => {
    expect(editLineageForResult(chain, 1)).toHaveLength(1);
  });

  it("reports nothing for a result with no input reference", () => {
    const generated = [
      message("user-1", null, [textPart("t1", "A harbor at dusk")]),
      message("assistant-1", "user-1", [imagePart("out-1", "artifact-fresh")]),
    ];
    expect(editLineageForResult(generated, 1)).toEqual([]);
  });

  it("stops where the producer is not in this transcript", () => {
    // A fork carries results, not history: the source's producer is absent.
    const forked = [
      message("user-1", null, [
        textPart("t1", "Brighten the sky"),
        imagePart("ref-1", "artifact-from-elsewhere", true),
      ]),
      message("assistant-1", "user-1", [imagePart("out-1", "artifact-bright")]),
    ];
    expect(editLineageForResult(forked, 1)).toEqual([
      { artifactId: "artifact-from-elsewhere", instruction: "Brighten the sky", messageId: "user-1" },
    ]);
  });

  it("never counts a preview as a producer", () => {
    const preview = imagePart("preview", "artifact-step-1");
    preview.metadata_json.preview = true;
    const withPreview = [
      message("assistant-0", null, [preview]),
      ...chain.slice(2),
    ];
    // The only "producer" of artifact-step-1 is a preview, so the chain is one step.
    expect(editLineageForResult(withPreview, 2)).toHaveLength(1);
  });
});
