/** Where a studio session's running edit has got to, read from the session itself. */

import { describe, expect, it } from "vitest";
import { studioApplyProgress } from "./studioApplyProgress";
import type { ChatDetail, Message, MessagePart } from "./types";

function part(type: string, text: string | null, metadata: Record<string, unknown> = {}): MessagePart {
  return { id: `${type}-${text}`, type, text, artifact_id: null, metadata_json: metadata } as unknown as MessagePart;
}

let made = 0;
function message(role: "user" | "assistant", status: string, parts: MessagePart[]): Message {
  made += 1;
  return { id: `${role}-${status}-${made}`, role, status, parts } as unknown as Message;
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

    expect(found?.part).toBe(newest);
    expect(found?.place).toBeNull();
  });

  it("is the first result of several still to come, and says which one it is", () => {
    const running = part("progress", "Sampling", { progress: 0.4, phase: "sampling" });
    const waiting = part("progress", "Queued", { progress: 0, phase: "queued", indeterminate: true });

    const found = studioApplyProgress(session([
      message("user", "complete", [part("text", "Make it warmer")]),
      message("assistant", "complete", [part("image", null)]),
      message("user", "complete", [part("text", "Make it cooler")]),
      message("assistant", "complete", [part("image", null)]),
      message("assistant", "pending", [running]),
      message("assistant", "pending", [waiting]),
    ]));

    // The earlier request's answer is not one of this edit's results.
    expect(found).toEqual({ part: running, place: { index: 2, count: 3 } });
  });

  it("counts a result that was stopped or failed among the several", () => {
    const running = part("progress", "Sampling", { progress: 0.1, phase: "sampling" });

    const found = studioApplyProgress(session([
      message("user", "complete", [part("text", "Make it cooler")]),
      message("assistant", "failed", [part("text", "")]),
      message("assistant", "pending", [running]),
    ]));

    expect(found?.place).toEqual({ index: 2, count: 2 });
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
