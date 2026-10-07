import type { InfiniteData, QueryClient } from "@tanstack/react-query";
import type { Chat, ChatMessageWindow, ChatTranscriptContext, TurnAccepted } from "./types";
import { chatMessageWindowPages, type ChatMessagePageCursor } from "./useChatMessagePages";

export function applyAcceptedChatPages(client: QueryClient, chatId: string, accepted: TurnAccepted, activate: boolean) {
  const current = client.getQueryData<Chat>(["chat", chatId]);
  const previousHead = current?.active_head_message_id ?? null;
  const outputs = accepted.assistant_messages?.length ? accepted.assistant_messages : [accepted.assistant_message];
  const head = outputs[outputs.length - 1].id;
  const continuing = accepted.user_message.parent_id === previousHead || head === previousHead;
  const key = ["chat", chatId, "messages", head];
  const prior = client.getQueryData<InfiniteData<ChatMessageWindow, ChatMessagePageCursor>>(
    continuing ? ["chat", chatId, "messages", previousHead] : key,
  );
  const messages = new Map([...(prior?.pages ?? [])].reverse()
    .flatMap((page) => page.messages).map((message) => [message.id, message]));
  for (const message of [accepted.user_message, ...outputs]) messages.set(message.id, message);
  const ordered = [...messages.values()];
  const older = prior?.pages.at(-1)?.has_older ?? Boolean(accepted.user_message.parent_id);
  client.setQueryData<InfiniteData<ChatMessageWindow, ChatMessagePageCursor>>(
    key, chatMessageWindowPages(chatId, ordered, older),
  );
  if (continuing) {
    const context = client.getQueryData<ChatTranscriptContext>(["chat", chatId, "context", previousHead]);
    if (context) {
      const parts = [accepted.user_message, ...outputs].flatMap((message) => message.parts);
      const visual = parts.filter((part) => part.artifact_id && part.metadata_json.preview !== true);
      client.setQueryData<ChatTranscriptContext>(["chat", chatId, "context", head], {
        ...context, head_id: head,
        has_prior_visual: context.has_prior_visual || visual.some((part) => ["image", "video"].includes(part.type)),
        has_prior_image: context.has_prior_image || visual.some((part) => part.type === "image"),
        has_pending_response: context.has_pending_response || outputs.some((message) => message.status === "pending"),
      });
    }
  }
  if (activate) client.setQueryData<Chat>(["chat", chatId], (chat) => chat ? {
    ...chat, active_head_message_id: head,
  } : chat);
}
