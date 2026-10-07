import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { MessageBubble } from "./MessageBubble";
import type { Message, MessagePart, ResponseRevision } from "./types";

const stamp = "2026-09-20T00:00:00Z";
const part: MessagePart = { id: "part", position: 0, type: "text", text: "Authoritative response", artifact_id: null, metadata_json: {} };
const revision: ResponseRevision = {
  id: "revision", message_id: "message", run_id: "run", sequence: 1, status: "complete", parts: [part],
  activity: { id: "activity", sequence: 3, message_id: "message", response_revision_id: "revision", occurred_at: stamp },
  created_at: stamp, updated_at: stamp,
};
const message: Message = {
  id: "message", chat_id: "chat", role: "assistant", status: "complete", parent_id: null,
  transcript_visible: true, content_removed_at: null, parts: [part], active_response_revision_id: "revision",
  response_revisions: [revision], created_at: stamp, updated_at: stamp,
};
afterEach(cleanup);

it("marks the authoritative revision content and ignores stale streaming text after completion", () => {
  const { container } = render(<MessageBubble message={message} liveText="Stale streamed prefix" />);
  expect(screen.getByText("Authoritative response")).toBeVisible();
  expect(screen.queryByText("Stale streamed prefix")).not.toBeInTheDocument();
  expect(JSON.parse(container.querySelector<HTMLElement>("[data-chat-activity]")!.dataset.chatActivity!)).toEqual(revision.activity);
});

it("does not attach acknowledgment to pending or removed content", () => {
  const { container, rerender } = render(<MessageBubble message={{ ...message, status: "pending" }} />);
  expect(container.querySelector("[data-chat-activity]")).toBeNull();
  rerender(<MessageBubble message={{ ...message, content_removed_at: stamp }} />);
  expect(container.querySelector("[data-chat-activity]")).toBeNull();
  expect(screen.getByText("Message removed")).toBeVisible();
});

it("renders a failed replacement and binds its acknowledgment to the failure", () => {
  const failed: ResponseRevision = { ...revision, id: "failed", sequence: 2, status: "failed", parts: [],
    activity: { ...revision.activity!, id: "failure-activity", sequence: 4, response_revision_id: "failed" } };
  render(<MessageBubble message={{ ...message, response_revisions: [revision, failed] }} />);
  const failure = screen.getByText("Another response failed.");
  expect(failure).toBeVisible();
  expect(JSON.parse(failure.closest<HTMLElement>("[data-chat-activity]")!.dataset.chatActivity!)).toEqual(failed.activity);
  expect(screen.getByText("Authoritative response")).toBeVisible();
});

it("shows retained cancelled replacement output while leaving the cancellation label unmarked", () => {
  const cancelled: ResponseRevision = { ...revision, id: "cancelled", sequence: 2, status: "cancelled",
    parts: [{ ...part, id: "partial", text: "Retained partial output" }],
    activity: { ...revision.activity!, id: "partial-activity", sequence: 4, response_revision_id: "cancelled" } };
  render(<MessageBubble message={{ ...message, response_revisions: [revision, cancelled] }} />);
  expect(screen.getByText("Another response was cancelled.").closest("[data-chat-activity]")).toBeNull();
  const partial = screen.getByText("Retained partial output");
  expect(partial).toBeVisible();
  expect(JSON.parse(partial.closest<HTMLElement>("[data-chat-activity]")!.dataset.chatActivity!)).toEqual(cancelled.activity);
});

it("does not attach an activity identity belonging to a different revision", () => {
  const { container } = render(<MessageBubble message={{ ...message,
    response_revisions: [{ ...revision, activity: { ...revision.activity!, response_revision_id: "other" } }],
  }} />);
  expect(container.querySelector("[data-chat-activity]")).toBeNull();
});
