import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { expect, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import type { ChatDetail, EditedBranch, Message } from "./types";

export async function exerciseEditedBranchNavigation() {
  const stamp = "2026-09-07T00:00:00Z";
  const message = (id: string, role: "user" | "assistant", text: string, parent: string | null): Message => ({
    id, chat_id: "branch-navigation", role, parent_id: parent, status: "complete",
    parts: [{ id: id + "-part", position: 0, type: "text", text, artifact_id: null, metadata_json: {} }],
    created_at: stamp, updated_at: stamp,
  });
  const original = message("source-user", "user", "Describe a blue boat", null);
  const response = message("source-answer", "assistant", "The blue boat floats.", original.id);
  const editedUser = message("edited-user", "user", "Describe a green boat", null);
  const editedAnswer = message("edited-answer", "assistant", "The green boat floats.", editedUser.id);
  let chat: ChatDetail = {
    id: "branch-navigation", project_id: null, title: "Boat versions", pinned: false, archived: false,
    routing_mode: "text", confirm_uncertain_media: false, active_chat_profile_id: null,
    active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: response.id,
    created_at: stamp, updated_at: stamp, messages: [original, response, editedUser, editedAnswer],
  };
  const branch: EditedBranch = {
    source_message_id: original.id, source_run_id: "source-run", branch_head_message_id: editedAnswer.id,
    source_available: true, can_continue: true, jobs: [],
    plan: { id: "edited-plan", chat_id: chat.id, idempotency_key: "edited-request",
      source_action: "edit_and_branch", persistence_scope: "durable", status: "complete",
      context_head_message_id: null, transcript_sequence: 2, priority: 0, planner_version: "single-turn-v1",
      failure_policy: "stop", summary_json: {}, steps: [], created_at: stamp, updated_at: stamp },
  };
  localStorage.setItem("local-lm-chat", chat.id);
  vi.mocked(api.chats).mockImplementation(async () => [chat]);
  vi.mocked(api.chat).mockImplementation(async () => chat);
  vi.mocked(api.editedBranches).mockResolvedValue({ items: [branch], next_cursor: null });
  vi.mocked(api.activateEditedBranch).mockImplementation(async (chatId, _planId, expectedHead) => {
    expect(expectedHead).toBe(response.id);
    chat = { ...chat, active_head_message_id: editedAnswer.id };
    return { chat_id: chatId, active_head_message_id: editedAnswer.id };
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
  const view = await screen.findByRole("button", { name: "View edited branch" });
  expect(screen.getByText("The blue boat floats.")).toBeVisible();
  expect(screen.queryByText("The green boat floats.")).not.toBeInTheDocument();
  fireEvent.click(view);
  const preview = await screen.findByRole("dialog", { name: "Edited branch preview" });
  expect(within(preview).getByText("The green boat floats.")).toBeVisible();
  expect(screen.getByText("The blue boat floats.")).toBeVisible();
  expect(api.activateEditedBranch).not.toHaveBeenCalled();
  expect(chat.active_head_message_id).toBe(response.id);
  fireEvent.click(within(preview).getByRole("button", { name: "Continue from this version" }));
  await waitFor(() => expect(screen.queryByRole("dialog", { name: "Edited branch preview" })).not.toBeInTheDocument());
  expect(api.activateEditedBranch).toHaveBeenCalledWith(chat.id, branch.plan.id, response.id);
  expect(screen.getByText("The green boat floats.")).toBeVisible();
  expect(screen.queryByText("The blue boat floats.")).not.toBeInTheDocument();
  expect(screen.getByText("Current version")).toBeVisible();
  expect(api.updateChat).not.toHaveBeenCalled();
}
