import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { expect, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import type { Chat, EngineCapabilities, SettingField } from "./types";

export async function exercisePriorTurnEditor(
  roleAwareMediaEngine: EngineCapabilities,
  contextLengthSetting: SettingField,
  maxTokensSetting: SettingField,
) {
    const stamp = "2026-07-22T00:00:00Z";
    const chat = {
      id: "chat-turn-overrides",
      project_id: null,
      title: "Turn overrides",
      pinned: false, archived: false,
      routing_mode: "text" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "assistant-turn-overrides",
      created_at: stamp,
      updated_at: stamp,
    };
    const userMessage = {
      id: "user-turn-overrides",
      chat_id: chat.id,
      parent_id: null,
      role: "user" as const,
      status: "complete" as const,
      parts: [{ id: "user-part", position: 0, type: "text" as const, text: "Count to 100", artifact_id: null, metadata_json: {} }],
      created_at: stamp,
      updated_at: stamp,
    };
    const assistantMessage = {
      id: "assistant-turn-overrides",
      chat_id: chat.id,
      parent_id: userMessage.id,
      role: "assistant" as const,
      status: "complete" as const,
      parts: [{ id: "assistant-part", position: 0, type: "text" as const, text: "1 2 3", artifact_id: null, metadata_json: {} }],
      created_at: stamp,
      updated_at: stamp,
    };
    localStorage.setItem("local-lm-chat", chat.id);
    vi.mocked(api.engines).mockResolvedValue([{
      ...roleAwareMediaEngine,
      roles: ["chat"],
      operations: ["text"],
      settings: [contextLengthSetting, maxTokensSetting],
      settings_by_role: { chat: [contextLengthSetting, maxTokensSetting] },
    }]);
    vi.mocked(api.chats).mockResolvedValue([chat]);
    let persistedChat: Chat = { ...chat };
    vi.mocked(api.chat).mockImplementation(async () => ({
      ...persistedChat,
      messages: [userMessage, assistantMessage],
    }));
    vi.mocked(api.updateChat).mockImplementation(async (_id, values) => {
      persistedChat = { ...persistedChat, ...values };
      return persistedChat;
    });
    vi.mocked(api.sendTurn).mockReturnValue(new Promise(() => {}));
    vi.mocked(api.getPriorTurnEditSource).mockResolvedValue({
      source_user_message_id: userMessage.id, source_run_id: "source-run", source_snapshot_sha256: "a".repeat(64),
      chat_id: chat.id, text: "Count to 100", mode: "text", operation: "text", input_artifact_ids: [], input_artifacts: [],
      references: [], settings: { max_tokens: 512 }, resolved_settings: { max_tokens: 512 }, settings_role: "chat",
      output_count: 1, profile_id: null, vision_profile_id: null, preset_id: null, preset: null, model_selection: {},
      workflow_selection: { selector_capability: "chat", mode: "legacy", workflow_family_id: null,
        workflow_revision_id: null, legacy_profile_id: null }, workflow_revision_id: null, workflow_schema: null,
      context_messages: [], context_visual_artifacts: [], profile_settings: {}, prompt_source: null,
    });
    vi.mocked(api.queueEditedMessage).mockRejectedValue(new Error("Queue full"));

    vi.mocked(api.regenerateMessage).mockReturnValue(new Promise(() => {}));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Turn settings" }));
    expect(screen.queryByRole("spinbutton", { name: /Context length/ })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("spinbutton", { name: /Maximum output/ }), { target: { value: "4096" } });
    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
    await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith(chat.id, {
      generation_settings_json: { chat: { max_tokens: 4096 } },
    }));

    fireEvent.click(screen.getByRole("button", { name: "Edit message" }));
    const editor = await screen.findByRole("dialog", { name: "Queue edited version" });
    const editedMessage = await within(editor).findByRole("textbox", { name: "Message" });
    fireEvent.change(editedMessage, { target: { value: "Count to 1000" } });
    fireEvent.click(within(editor).getByRole("button", { name: "Turn settings" }));
    expect(screen.getByRole("spinbutton", { name: /Maximum output/ })).toHaveValue(512);
    fireEvent.change(screen.getByRole("spinbutton", { name: /Maximum output/ }), { target: { value: "1536" } });
    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
    fireEvent.click(within(editor).getByRole("button", { name: "Queue edited version" }));
    await within(editor).findByText("Queue full");
    expect(api.queueEditedMessage).toHaveBeenCalledWith(userMessage.id, expect.objectContaining({
      text: "Count to 1000", mode: "text", settings: { max_tokens: 1536 }, idempotency_key: expect.any(String),
      source_run_id: "source-run", source_snapshot_sha256: "a".repeat(64),
    }));
    expect(api.branchMessage).not.toHaveBeenCalled();
    const firstRequest = vi.mocked(api.queueEditedMessage).mock.calls[0][1];
    fireEvent.click(within(editor).getByRole("button", { name: "Queue edited version" }));
    await waitFor(() => expect(api.queueEditedMessage).toHaveBeenCalledTimes(2));
    expect(vi.mocked(api.queueEditedMessage).mock.calls[1][1]).toEqual(firstRequest);
    await waitFor(() => expect(within(editor).getByRole("button", { name: "Queue edited version" })).toBeEnabled());
    fireEvent.click(within(editor).getByRole("button", { name: "Close edited version" }));
    expect(persistedChat.generation_settings_json).toEqual({ chat: { max_tokens: 4096 } });

    fireEvent.click(screen.getByRole("button", { name: "Regenerate response" }));
    await waitFor(() => expect(api.regenerateMessage).toHaveBeenCalledWith(assistantMessage.id, { max_tokens: 4096 }, expect.any(String)));


    // Deleting a turn is two-step: the intent button, then a confirmation
    // that names what else goes with it.
    vi.mocked(api.deleteExchange).mockResolvedValue({
      chat_id: chat.id,
      user_message_id: userMessage.id,
      message_ids: [userMessage.id, assistantMessage.id],
      run_ids: [],
      job_ids: [],
      work_plan_ids: [],
      released_artifact_ids: [],
      retained_artifact_ids: [],
      new_head_message_id: null,
    });
    fireEvent.click(screen.getByRole("button", { name: "Delete this turn" }));
    expect(api.deleteExchange).not.toHaveBeenCalled();
    expect(screen.getByText("Also deletes the answer and its media.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Delete turn" }));
    await waitFor(() => expect(api.deleteExchange).toHaveBeenCalledWith(userMessage.id));

    fireEvent.change(screen.getByRole("textbox", { name: "Message" }), { target: { value: "Count to 1000" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(api.sendTurn).toHaveBeenCalledWith(
      chat.id, "Count to 1000", "text", [], { max_tokens: 4096 }, expect.any(String), "turns", undefined, [], undefined, undefined, expect.any(Function),
    ));

}
