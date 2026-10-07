import { vi } from "vitest";
import { api } from "./api";
import { editLineageForResult } from "./messageMedia";
import type { ChatDetail, Message } from "./types";

/** Project existing constructed chat fixtures onto the bounded read contracts. */
export function installChatReadFixtures() {
  const latest = new Map<string, Promise<ChatDetail>>();
  const read = (id: string) => latest.get(id) ?? api.chat(id);
  const branch = (detail: ChatDetail, head?: string | null): Message[] => {
    if (!head) return detail.messages;
    const byId = new Map(detail.messages.map((message) => [message.id, message]));
    const ids = new Set<string>();
    let message = byId.get(head);
    while (message && !ids.has(message.id)) {
      ids.add(message.id);
      message = message.parent_id ? byId.get(message.parent_id) : undefined;
    }
    return detail.messages.filter((item) => ids.has(item.id));
  };
  api.chatMetadata = vi.fn(async (id: string) => {
    const request = api.chat(id);
    latest.set(id, request);
    const detail = await request;
    return Object.fromEntries(Object.entries(detail).filter(([key]) =>
      key !== "messages" && key !== "web_searches")) as Omit<ChatDetail, "messages">;
  });
  api.chatMessages = vi.fn(async (id: string, options: Parameters<typeof api.chatMessages>[1] = {}) => {
    const all = branch(await read(id), options.headId);
    const limit = options.limit ?? 40;
    let start = Math.max(0, all.length - limit);
    let end = all.length;
    if (options.before) { end = all.findIndex((message) => message.id === options.before); start = Math.max(0, end - limit); }
    if (options.after) { start = all.findIndex((message) => message.id === options.after) + 1; end = start + limit; }
    if (options.around) { start = Math.max(0, all.findIndex((message) => message.id === options.around) - Math.floor((limit - 1) / 2)); end = start + limit; }
    return { chat_id: id, messages: all.slice(start, end), has_older: start > 0, has_newer: end < all.length };
  });
  api.chatEditLineage = vi.fn(async (id: string, resultId: string, options: Parameters<typeof api.chatEditLineage>[2] = {}) => {
    const messages = branch(await read(id), resultId);
    const all = editLineageForResult(messages, messages.findIndex((message) => message.id === resultId)).reverse();
    const start = options.before ? all.findIndex((step) => step.messageId === options.before) + 1 : 0;
    const steps = all.slice(start, start + (options.limit ?? 40));
    return { chat_id: id, result_message_id: resultId, steps: steps.map((step) => ({
      message_id: step.messageId, artifact_id: step.artifactId, instruction: step.instruction,
    })), next_before: start + steps.length < all.length ? steps.at(-1)!.messageId : null };
  });
  api.chatContext = vi.fn(async (id: string, head: string | null) => {
    const messages = branch(await read(id), head).filter((message) => message.transcript_visible !== false && !message.content_removed_at);
    const parts = messages.flatMap((message) => message.parts).filter((part) => part.artifact_id && part.metadata_json.preview !== true);
    return { chat_id: id, head_id: head, has_prior_visual: parts.some((part) => ["image", "video"].includes(part.type)),
      has_prior_image: parts.some((part) => part.type === "image"),
      has_pending_response: messages.some((message) => message.status === "pending" || message.response_revisions?.some((revision) => revision.status === "pending")) };
  });
  api.chatSearches = vi.fn(async (id: string, options: Parameters<typeof api.chatSearches>[1] = {}) => {
    const detail = await read(id);
    const messages = branch(detail, options.headId);
    const oldest = options.oldestMessageId ? messages.findIndex((message) => message.id === options.oldestMessageId) : 0;
    const ids = new Set(messages.slice(Math.max(0, oldest)).map((message) => message.id));
    const all = (detail.web_searches ?? []).filter((search) => options.pendingOnly
      ? search.job_id !== null && ["awaiting_approval", "approved", "scheduled"].includes(search.state)
      : ids.has(search.assistant_message_id));
    const end = options.before ? all.findIndex((search) => search.run_id === options.before) : all.length;
    const start = Math.max(0, end - (options.limit ?? 40));
    return { chat_id: id, searches: all.slice(start, end), next_before: start > 0 ? all[start].run_id : null };
  });
}
