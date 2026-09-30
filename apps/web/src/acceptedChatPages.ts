import type { InfiniteData, QueryClient } from "@tanstack/react-query";
import type { Chat, ChatMessageWindow, ChatTranscriptContext, TurnAccepted } from "./types";
import type { ChatMessagePageCursor } from "./useChatMessagePages";

export function applyAcceptedChatPages(client: QueryClient, chatId: string, accepted: TurnAccepted, activate: boolean) {
  const current = client.getQueryData<Chat>(["chat", chatId]);
  const previousHead = current?.active_head_message_id ?? null;
  const head = accepted.assistant_message.id;
  const continuing = accepted.user_message.parent_id === previousHead || head === previousHead;
  const key = ["chat", chatId, "messages", head];
  const prior = client.getQueryData<InfiniteData<ChatMessageWindow, ChatMessagePageCursor>>(
    continuing ? ["chat", chatId, "messages", previousHead] : key,
  );
  const messages = new Map([...(prior?.pages ?? [])].reverse()
    .flatMap((page) => page.messages).map((message) => [message.id, message]));
  for (const message of [accepted.user_message, accepted.assistant_message]) messages.set(message.id, message);
  const ordered = [...messages.values()];
  const older = prior?.pages.at(-1)?.has_older ?? Boolean(accepted.user_message.parent_id);
  const pages: ChatMessageWindow[] = [];
  for (let end = ordered.length; end > 0; end -= 40) {
    const start = Math.max(0, end - 40);
    pages.push({ chat_id: chatId, messages: ordered.slice(start, end),
      has_older: start > 0 || older, has_newer: end < ordered.length });
  }
  client.setQueryData<InfiniteData<ChatMessageWindow, ChatMessagePageCursor>>(key, {
    pages, pageParams: pages.map((page, index) => ({
      before: index ? pages[index - 1].messages[0].id : undefined,
      limit: index ? page.messages.length : 40,
    })),
  });
  if (continuing) {
    const context = client.getQueryData<ChatTranscriptContext>(["chat", chatId, "context", previousHead]);
    if (context) {
      const parts = [accepted.user_message, accepted.assistant_message].flatMap((message) => message.parts);
      const visual = parts.filter((part) => part.artifact_id && part.metadata_json.preview !== true);
      client.setQueryData<ChatTranscriptContext>(["chat", chatId, "context", head], {
        ...context, head_id: head,
        has_prior_visual: context.has_prior_visual || visual.some((part) => ["image", "video"].includes(part.type)),
        has_prior_image: context.has_prior_image || visual.some((part) => part.type === "image"),
        has_pending_response: context.has_pending_response || accepted.assistant_message.status === "pending",
      });
    }
  }
  if (activate) client.setQueryData<Chat>(["chat", chatId], (chat) => chat ? {
    ...chat, active_head_message_id: head,
  } : chat);
}
