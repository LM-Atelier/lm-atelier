import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { MessageBubble } from "./MessageBubble";
import { UserMessageControls } from "./MessageRemovalControls";
import type { Message } from "./types";

afterEach(cleanup);

const stamp = "2026-10-07T12:00:00Z";
const message: Message = {
  id: "msg-response", chat_id: "chat-response", parent_id: null,
  role: "assistant", status: "complete", transcript_visible: true,
  content_removed_at: null, active_response_revision_id: null,
  parts: [{ id: "part-response", position: 0, type: "text", text: "A completed response.",
    artifact_id: null, metadata_json: {} }],
  references: [], response_revisions: [], feedback: null, created_at: stamp, updated_at: stamp,
};

it("keeps the focused thread action in place and prevents starting a thread during a reply", () => {
  const fork = vi.fn();
  const { rerender } = render(<MessageBubble message={message} onForkThread={fork} />);
  const button = screen.getByRole("button", { name: "Start a new thread here" });
  button.focus();
  rerender(<MessageBubble message={message} onForkThread={fork} threadActionsDisabled />);
  expect(screen.getByRole("button", { name: "Start a new thread here" })).toBe(button);
  expect(button).toHaveFocus();
  expect(button).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(button);
  expect(fork).not.toHaveBeenCalled();
  rerender(<MessageBubble message={message} onForkThread={fork} threadActionsDisabled={false} />);
  fireEvent.click(button);
  expect(fork).toHaveBeenCalledExactlyOnceWith(message.id);
});

it("does not open content removal while conversation actions are disabled", () => {
  const remove = vi.fn();
  render(<MessageBubble message={message} onRemoveItem={remove} threadActionsDisabled />);
  fireEvent.click(screen.getByRole("button", { name: "Remove this item, keep replies" }));
  expect(screen.queryByText("Only this item's content is removed. Replies stay.")).not.toBeInTheDocument();
  expect(remove).not.toHaveBeenCalled();
});

it("keeps an open turn deletion confirmation inert when a reply begins", () => {
  const remove = vi.fn();
  const props = { messageId: "msg-input", createdAt: stamp, copyableText: "An earlier input.", onDeleteExchange: remove };
  const { rerender } = render(<UserMessageControls {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Delete this turn" }));
  const confirm = screen.getByRole("button", { name: /^Delete turn$/ });
  confirm.focus();
  rerender(<UserMessageControls {...props} disabled />);
  expect(confirm).toHaveFocus();
  fireEvent.click(confirm);
  expect(remove).not.toHaveBeenCalled();
  expect(confirm).toBeInTheDocument();
  rerender(<UserMessageControls {...props} disabled={false} />);
  fireEvent.click(confirm);
  expect(remove).toHaveBeenCalledExactlyOnceWith(props.messageId);
});

it("keeps an open content removal confirmation inert when a reply begins", () => {
  const remove = vi.fn();
  const props = { messageId: "msg-input", createdAt: stamp, copyableText: "An earlier input.", onRemoveItem: remove };
  const { rerender } = render(<UserMessageControls {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Remove this item, keep replies" }));
  const confirm = screen.getByRole("button", { name: "Remove this item, keep replies" });
  confirm.focus();
  rerender(<UserMessageControls {...props} disabled />);
  expect(confirm).toHaveFocus();
  fireEvent.click(confirm);
  expect(remove).not.toHaveBeenCalled();
  expect(confirm).toBeInTheDocument();
  rerender(<UserMessageControls {...props} disabled={false} />);
  fireEvent.click(confirm);
  expect(remove).toHaveBeenCalledExactlyOnceWith(props.messageId);
});
