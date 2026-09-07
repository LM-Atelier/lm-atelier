import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MessageBubble } from "./App";
import type { Message } from "./types";

afterEach(cleanup);

describe("MessageBubble content tombstones", () => {
  it("renders only the removal marker even if stale payload or live text arrives", () => {
    const stamp = "2026-08-25T10:00:00Z";
    const removed: Message = {
      id: "msg-tombstone",
      chat_id: "chat-tombstone",
      parent_id: null,
      role: "user",
      status: "complete",
      transcript_visible: true,
      content_removed_at: stamp,
      active_response_revision_id: null,
      parts: [
        {
          id: "part-must-not-render",
          position: 0,
          type: "text",
          text: "stored payload must not render",
          artifact_id: null,
          metadata_json: {},
        },
      ],
      references: [],
      response_revisions: [],
      feedback: null,
      created_at: stamp,
      updated_at: stamp,
    };

    render(
      <MessageBubble
        message={removed}
        liveText="streamed payload must not render"
        onEdit={vi.fn()}
        onDeleteExchange={vi.fn()}
        onForkThread={vi.fn()}
      />,
    );

    expect(screen.getByText("Message removed")).toBeVisible();
    expect(screen.queryByText("stored payload must not render")).not.toBeInTheDocument();
    expect(screen.queryByText("streamed payload must not render")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit message" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Copy user message" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete this turn" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start a new thread here" })).not.toBeInTheDocument();
  });
});

describe("MessageBubble edit review status", () => {
  const stamp = "2026-09-07T10:00:00Z";

  const assistant = (provenance: Record<string, unknown>): Message => ({
    id: "msg-review",
    chat_id: "chat-review",
    parent_id: null,
    role: "assistant",
    status: "complete",
    transcript_visible: true,
    content_removed_at: null,
    active_response_revision_id: null,
    parts: [
      {
        id: "part-text",
        position: 0,
        type: "text",
        text: "Here is the edited picture.",
        artifact_id: null,
        metadata_json: {},
      },
      {
        id: "part-metadata",
        position: 1,
        type: "generation_metadata",
        text: null,
        artifact_id: null,
        metadata_json: { provenance },
      },
    ],
    references: [],
    response_revisions: [],
    feedback: null,
    created_at: stamp,
    updated_at: stamp,
  });

  it("says a second image was an automatic retry, and which way strength moved", () => {
    render(<MessageBubble message={assistant({
      image_edit_verification_retry: {
        source_run_id: "run-source",
        strength_before: 0.35,
        strength_after: 0.47,
      },
    })} />);

    expect(
      screen.getByText("Automatic retry after edit review at a higher strength"),
    ).toBeVisible();
  });

  it("says on the original result that the review retried it", () => {
    render(<MessageBubble message={assistant({
      image_edit_verification: {
        version: "image-edit-verification-v1",
        status: "complete",
        automatic_retry_executed: true,
        assessment: {
          requested_change_visible: false,
          unrelated_content_preserved: true,
          retry_recommended: true,
          direction: "increase",
          confidence: 0.82,
        },
        strength_adjustment: {
          parameter: "denoise",
          before: 0.4,
          after: 0.52,
          bounds: { minimum: 0.05, maximum: 0.95 },
        },
      },
    })} />);

    expect(screen.getByText("Edit review retried this once at a higher strength")).toBeVisible();
  });

  it("says nothing about a message the review never looked at", () => {
    render(<MessageBubble message={assistant({
      model_selection: { mode: "manual" },
    })} />);

    expect(screen.queryByText(/edit review/i)).not.toBeInTheDocument();
  });
});
