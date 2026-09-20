import type { Chat, ChatSummary } from "./types";

export function asChatSummary(chat: Chat): ChatSummary {
  return {
    id: chat.id, project_id: chat.project_id, title: chat.title, archived: chat.archived,
    pinned: chat.pinned, created_at: chat.created_at, updated_at: chat.updated_at,
    activity: { active_work_count: 0, unresolved_failed_count: 0, last_output: null, last_failure: null },
  };
}
