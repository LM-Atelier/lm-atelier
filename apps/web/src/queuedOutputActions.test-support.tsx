import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { expect, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import type { ChatDetail, EngineCapabilities, Message } from "./types";

export async function exerciseQueuedOutputActions(
  engines: EngineCapabilities,
  pending: "other-message" | "other-revision" | "target-revision",
) {
  const stamp = "2026-07-22T00:00:00Z";
  const output: Message = {
    id: "completed-image", chat_id: "queued-output-actions", parent_id: null,
    role: "assistant", status: "complete", created_at: stamp, updated_at: stamp,
    parts: [{ id: "image-part", position: 0, type: "image", text: null,
      artifact_id: "sha256:completed-source", metadata_json: {} }],
  };
  const later: Message = {
    ...output, id: "later-answer", parent_id: output.id,
    status: pending === "other-message" ? "pending" : "complete",
    parts: [{ id: "later-text", position: 0, type: "text", text: "Later answer",
      artifact_id: null, metadata_json: {} }],
  };
  if (pending !== "other-message") {
    const target = pending === "target-revision" ? output : later;
    target.response_revisions = [{
      id: "pending-revision", message_id: target.id, sequence: 2, status: "pending",
      parts: [], run_id: "pending-run", created_at: stamp, updated_at: stamp,
    }];
  }
  const chat: ChatDetail = {
    id: output.chat_id, project_id: null, title: "Queued output actions",
    pinned: false, archived: false, routing_mode: "image", confirm_uncertain_media: false,
    active_chat_profile_id: null, active_image_profile_id: null, active_video_profile_id: null,
    active_head_message_id: later.id, created_at: stamp, updated_at: stamp,
    messages: [output, later],
  };
  localStorage.setItem("local-lm-chat", chat.id);
  vi.mocked(api.chats).mockResolvedValue([chat]);
  vi.mocked(api.chat).mockResolvedValue(chat);
  vi.mocked(api.engines).mockResolvedValue([engines]);
  vi.mocked(api.regenerateMessage).mockReturnValue(new Promise(() => {}));
  vi.mocked(api.updateChat).mockImplementation(async (_id, values) => ({ ...chat, ...values }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);

  fireEvent.click(await screen.findByRole("button", { name: "Reference this media" }));
  expect(await screen.findByRole("link", { name: "Preview sha256:completed-source" })).toBeVisible();
  expect(screen.getByRole("button", { name: "Edit this image" })).toBeEnabled();
  expect(screen.getByRole("button", { name: "Animate this image" })).toBeEnabled();
  expect(screen.getByRole("button", { name: "Open this image in the Image Studio" })).toBeEnabled();
  expect(api.cancelChat).not.toHaveBeenCalled();
  expect(api.cancelWorkPlan).not.toHaveBeenCalled();
  const regenerate = screen.getAllByRole("button", { name: "Regenerate response" });
  expect(regenerate).toHaveLength(1);
  fireEvent.click(regenerate[0]);
  await waitFor(() => expect(api.regenerateMessage).toHaveBeenCalledWith(
    pending === "target-revision" ? later.id : output.id, {}, expect.any(String),
  ));
  expect(api.cancelChat).not.toHaveBeenCalled();
}
