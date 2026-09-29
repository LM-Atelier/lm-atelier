/** Where a studio session's running edit has got to, read from the session itself. */

import { describe, expect, it } from "vitest";
import { studioApplyProgress } from "./studioApplyProgress";
import type { ChatDetail, Message, MessagePart } from "./types";

function part(type: string, text: string | null, metadata: Record<string, unknown> = {}): MessagePart {
  return { id: `${type}-${text}`, type, text, artifact_id: null, metadata_json: metadata } as unknown as MessagePart;
}

function message(role: "user" | "assistant", status: string, parts: MessagePart[]): Message {
  return { id: `${role}-${status}-${parts.length}`, role, status, parts } as unknown as Message;
}

function session(messages: Message[]): ChatDetail {
  return { messages } as unknown as ChatDetail;
}

describe("the running edit's progress", () => {
  it("is the progress of the newest answer still being made", () => {
    const earlier = part("progress", "Queued", { progress: 0, phase: "queued", indeterminate: true });
    const newest = part("progress", "Sampling", { progress: 0.4, phase: "sampling" });

    const found = studioApplyProgress(session([
      message("user", "complete", [part("text", "Make it warmer")]),
      message("assistant", "pending", [earlier]),
      message("user", "complete", [part("text", "Make it cooler")]),
      message("assistant", "pending", [newest, part("image", null, { preview: true })]),
    ]));

    expect(found).toBe(newest);
  });

  it("is nothing once every answer has finished, or before there is a session", () => {
    const finished = session([
      message("user", "complete", [part("text", "Make it warmer")]),
      message("assistant", "complete", [part("image", null)]),
    ]);

    expect(studioApplyProgress(finished)).toBeNull();
    expect(studioApplyProgress(undefined)).toBeNull();
  });

  it("is nothing for a running answer that reports no progress", () => {
    expect(studioApplyProgress(session([message("assistant", "pending", [part("text", "")])]))).toBeNull();
  });
});
