/** Where a studio session's running edit has got to, read from the session itself. */

import { describe, expect, it } from "vitest";
import { studioApplyFailure, studioApplyProgress } from "./studioApplyProgress";
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

describe("why the newest edit came back without a picture", () => {
  it("gives the reason the server recorded on the failed answer", () => {
    const request = message("user", "complete", [part("text", "Make it warmer")]);

    const found = studioApplyFailure(session([
      request,
      message("assistant", "failed", [part("error", " The workflow stopped at its sampler. ")]),
    ]));

    expect(found).toEqual({ requestId: request.id, text: "The edit did not finish: The workflow stopped at its sampler." });
  });

  it("counts the results that did not finish among several, while the rest are still made", () => {
    const found = studioApplyFailure(session([
      message("user", "complete", [part("text", "Make it cooler")]),
      message("assistant", "complete", [part("image", null)]),
      message("assistant", "failed", [part("error", "Out of memory")]),
      message("assistant", "pending", [part("progress", "Sampling")]),
    ]));

    expect(found?.text).toBe("1 of 3 results did not finish: Out of memory");
  });

  it("says only that it did not finish when no reason was recorded", () => {
    const found = studioApplyFailure(session([
      message("user", "complete", [part("text", "Make it cooler")]),
      message("assistant", "failed", [part("error", "  ")]),
    ]));

    expect(found?.text).toBe("The edit did not finish.");
  });

  it("is nothing once a newer edit is asked for, for an edit that finished or was stopped, or before a session", () => {
    const failedEarlier = [
      message("user", "complete", [part("text", "Make it warmer")]),
      message("assistant", "failed", [part("error", "Out of memory")]),
    ];

    expect(studioApplyFailure(session([...failedEarlier, message("user", "complete", [part("text", "Try again")])]))).toBeNull();
    expect(studioApplyFailure(session([
      ...failedEarlier,
      message("user", "complete", [part("text", "Make it cooler")]),
      message("assistant", "pending", [part("progress", "Queued")]),
    ]))).toBeNull();
    expect(studioApplyFailure(session([
      message("user", "complete", [part("text", "Make it warmer")]),
      message("assistant", "complete", [part("image", null)]),
    ]))).toBeNull();
    expect(studioApplyFailure(session([
      message("user", "complete", [part("text", "Make it warmer")]),
      message("assistant", "cancelled", []),
    ]))).toBeNull();
    expect(studioApplyFailure(undefined)).toBeNull();
    expect(studioApplyFailure(session([]))).toBeNull();
  });
});
