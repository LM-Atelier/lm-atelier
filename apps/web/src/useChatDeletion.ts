import { useMutation, type QueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { withoutComposerDraft, type ComposerDraft } from "./composerPromptSource";
import { rememberCurrentChat } from "./currentChat";
import type { Chat, ChatSummary } from "./types";
import { changeChatPages, restoreChatPages, snapshotChatPages } from "./useChatPages";

/** Delete a chat, taking it off the list before the server answers.

If it was the open chat, the next chat that is not archived opens in its place
and is remembered. A refusal puts back the list and whichever chat was open.
Success clears what this browser still held for the chat, its unsaved field
edits and its composer draft, and refreshes the media and jobs it may have
owned. */
export function useChatDeletion({
  client,
  chats,
  currentChatId,
  activeChatId,
  setCurrentChatId,
  setChatDrafts,
  setComposerDrafts,
}: {
  client: QueryClient;
  /** The chat list as loaded now; the next open chat is chosen from it. */
  chats: readonly ChatSummary[] | undefined;
  currentChatId: string | null;
  activeChatId: string | null;
  setCurrentChatId: (chatId: string | null) => void;
  setChatDrafts: (update: (current: Record<string, Partial<Chat>>) => Record<string, Partial<Chat>>) => void;
  setComposerDrafts: (update: (current: Record<string, ComposerDraft>) => Record<string, ComposerDraft>) => void;
}) {
  return useMutation({
    mutationFn: ({ id, deleteGeneratedMedia }: { id: string; deleteGeneratedMedia: boolean }) => api.deleteChat(id, deleteGeneratedMedia),
    onMutate: async ({ id: deletedId }) => {
      await client.cancelQueries({ queryKey: ["chats"] });
      const previousChats = snapshotChatPages(client);
      const remainingChats = (chats ?? []).filter((candidate) => candidate.id !== deletedId);
      const previousCurrentChatId = currentChatId;
      changeChatPages(client, (item) => item.id === deletedId ? null : item);
      if (activeChatId === deletedId) {
        const nextChatId = remainingChats.find((candidate) => !candidate.archived)?.id ?? null;
        setCurrentChatId(nextChatId);
        rememberCurrentChat(nextChatId);
      }
      client.removeQueries({ queryKey: ["chat", deletedId], exact: true });
      return { previousChats, previousCurrentChatId };
    },
    onSuccess: (_value, { id: deletedId }) => {
      setChatDrafts((current) => {
        const next = { ...current };
        delete next[deletedId];
        return next;
      });
      setComposerDrafts((current) => withoutComposerDraft(current, deletedId));
      void client.invalidateQueries({ queryKey: ["artifacts"] });
      void client.invalidateQueries({ queryKey: ["artifact-storage"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (_error, _deletedChat, context) => {
      if (!context) return;
      restoreChatPages(client, context.previousChats);
      setCurrentChatId(context.previousCurrentChatId);
      rememberCurrentChat(context.previousCurrentChatId);
    },
    onSettled: () => void client.invalidateQueries({ queryKey: ["chats"] }),
  });
}
