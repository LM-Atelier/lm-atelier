import { api } from "./api";

/**
 * Remove a chat made for one start that was then refused, while it is still blank.
 *
 * Deleting a chat otherwise moves it to Recently Deleted, which would keep a
 * chat nobody saw. This goes through the removal of empty chats instead, for
 * this one chat only, and only while it is strictly blank: a chat with
 * anything in it, or one its settings set apart, is left as it is. It never
 * fails: a chat it could not remove stays blank, and empty-chat cleanup can
 * still remove it later.
 */
export async function discardBlankChat(chatId: string): Promise<void> {
  const choice = {
    chat_ids: [chatId],
    // Made moments ago, so no age keeps it.
    min_age_hours: 0,
    include_archived: false,
    include_configured: false,
  };
  try {
    const preview = await api.previewEmptyChats(choice);
    if (preview.strict_count !== 1 || preview.configured_count !== 0 || preview.conflicts.length > 0) {
      return;
    }
    await api.deleteEmptyChats({
      ...choice,
      operation_id: crypto.randomUUID(),
      preview_id: preview.preview_id,
      digest: preview.digest,
      acknowledged_count: 1,
      acknowledged_configured: false,
    });
  } catch {
    // Left as it is; the start's own failure is what the person is told.
  }
}
