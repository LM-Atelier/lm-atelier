/** Which chat was open last, so a reload opens it again. Kept in this browser. */
export const CURRENT_CHAT_KEY = "local-lm-chat";

/** Remember the open chat, or forget it when no chat is open. */
export function rememberCurrentChat(chatId: string | null): void {
  if (chatId) localStorage.setItem(CURRENT_CHAT_KEY, chatId);
  else localStorage.removeItem(CURRENT_CHAT_KEY);
}
