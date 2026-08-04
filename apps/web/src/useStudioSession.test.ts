import { describe, expect, it, vi } from "vitest";
import { studioSteps } from "./useStudioSession";
import type { ChatDetail, Message, MessagePart } from "./types";

const stamp = "2026-08-03T00:00:00Z";

function part(overrides: Partial<MessagePart>): MessagePart {
  return {
    id: "part",
    position: 0,
    type: "text",
    text: null,
    artifact_id: null,
    metadata_json: {},
    ...overrides,
  };
}

function message(overrides: Partial<Message> & { id: string }): Message {
  return {
    chat_id: "chat-studio",
    parent_id: null,
    role: "assistant",
    status: "complete",
    parts: [],
    created_at: stamp,
    updated_at: stamp,
    ...overrides,
  };
}

function session(messages: Message[]): ChatDetail {
  return {
    id: "chat-studio",
    project_id: null,
    title: "Studio",
    pinned: false,
  archived: true,
    routing_mode: "image",
    confirm_uncertain_media: false,
    active_chat_profile_id: null,
    active_image_profile_id: null,
    active_video_profile_id: null,
    active_head_message_id: null,
    created_at: stamp,
    updated_at: stamp,
    messages,
  };
}

describe("studio filmstrip", () => {
  it("starts at the source and adds one entry per result with its instruction", () => {
    const steps = studioSteps(
      session([
        message({
          id: "user-1",
          role: "user",
          parts: [part({ id: "t1", type: "text", text: "make it a watercolor" })],
        }),
        message({
          id: "answer-1",
          parts: [part({ id: "i1", type: "image", artifact_id: "art-1" })],
        }),
        message({
          id: "user-2",
          role: "user",
          parts: [part({ id: "t2", type: "text", text: "warmer light" })],
        }),
        message({
          id: "answer-2",
          parts: [part({ id: "i2", type: "image", artifact_id: "art-2" })],
        }),
      ]),
      "art-source",
    );

    expect(steps.map((step) => step.artifactId)).toEqual(["art-source", "art-1", "art-2"]);
    expect(steps[0].isSource).toBe(true);
    expect(steps[1].instruction).toBe("make it a watercolor");
    expect(steps[2].instruction).toBe("warmer light");
  });

  it("ignores previews and answers that produced no image", () => {
    const preview = part({ id: "p", type: "image", artifact_id: "art-preview" });
    preview.metadata_json = { preview: true };
    const steps = studioSteps(
      session([
        message({ id: "answer-preview", parts: [preview] }),
        message({ id: "answer-text", parts: [part({ id: "t", type: "text", text: "hm" })] }),
      ]),
      "art-source",
    );

    expect(steps.map((step) => step.artifactId)).toEqual(["art-source"]);
  });

  it("works before any edit and without a known source", () => {
    expect(studioSteps(session([]), "art-source")).toHaveLength(1);
    expect(studioSteps(session([]), null)).toEqual([]);
  });
});

describe("studio mask upload", () => {
  it("keeps a selection out of the turn's attachments", async () => {
    // A mask is instruction, not content: it must ride in settings so it
    // never renders as an attachment or counts toward edit lineage.
    const { api } = await import("./api");
    const upload = vi.spyOn(api, "upload").mockResolvedValue({ id: "sha256:mask" } as never);
    const sendTurn = vi.spyOn(api, "sendTurn").mockResolvedValue({} as never);

    const { uploadMaskForTest } = await import("./useStudioSession");
    const setting = await uploadMaskForTest({
      blob: new Blob([new Uint8Array([1, 2])], { type: "image/png" }),
      featherPx: 6,
      invert: true,
    });

    expect(upload).toHaveBeenCalled();
    expect(setting).toEqual({ artifact_id: "sha256:mask", feather_px: 6, invert: true });
    expect(sendTurn).not.toHaveBeenCalled();
    upload.mockRestore();
    sendTurn.mockRestore();
  });
});
