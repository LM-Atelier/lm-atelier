import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MessageBubble } from "./App";
import type { Message } from "./types";

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
