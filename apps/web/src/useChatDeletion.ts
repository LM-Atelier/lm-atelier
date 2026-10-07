import { useMutation, type QueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api, ApiError } from "./api";
import { rememberCurrentChat } from "./currentChat";
import type { ChatSummary } from "./types";
import type { RecoveryCommand, RecoveryItem } from "./recoveryTypes";
import { changeChatPages, restoreChatPages, snapshotChatPages } from "./useChatPages";

/** Move a checked chat to recovery and undo with its original identity.
 * A refusal restores the list. Local drafts survive while the chat is recoverable.
 */
export function useChatDeletion({
  client,
  chats,
  currentChatId,
  activeChatId,
  setCurrentChatId,
}: {
  client: QueryClient;
  /** The chat list as loaded now; the next open chat is chosen from it. */
  chats: readonly ChatSummary[] | undefined;
  currentChatId: string | null;
  activeChatId: string | null;
  setCurrentChatId: (chatId: string | null) => void;
}) {
  const [deleted, setDeleted] = useState<{ item: RecoveryItem; wasOpen: boolean; replacementId: string | null } | null>(null);
  const undoCommand = useRef<{ deletionId: string; command: RecoveryCommand } | null>(null);
  const undo = useMutation({
    mutationFn: async (item: RecoveryItem) => {
      if (undoCommand.current?.deletionId !== item.deletion_id) {
        const impact = await api.recoveryImpact(item.deletion_id);
        if (!impact.available_actions.includes("restore"))
          throw new ApiError(409, undefined, "This chat cannot be restored now. Check Recently Deleted in Settings.", "recovery-restore-unavailable");
        undoCommand.current = { deletionId: item.deletion_id, command: {
          expected_revision: impact.revision, impact_sha256: impact.impact_sha256, operation_key: crypto.randomUUID(),
        } };
      }
      return api.restoreRecovery(item.deletion_id, { ...undoCommand.current.command, restore_unfiled: false });
    },
    onSuccess: (result) => {
      if (deleted?.item.deletion_id === result.deletion_id && deleted.wasOpen && currentChatId === deleted.replacementId) {
        setCurrentChatId(result.subject_id);
        rememberCurrentChat(result.subject_id);
      }
      setDeleted((current) => current?.item.deletion_id === result.deletion_id ? null : current);
      for (const key of ["chats", "recovery-items", "projects", "empty-chats", "jobs"])
        void client.invalidateQueries({ queryKey: [key] });
      void client.invalidateQueries({ queryKey: ["chat", result.subject_id] });
    },
  });
  const mutation = useMutation({
    mutationFn: ({ id, deleteGeneratedMedia, command }: { id: string; deleteGeneratedMedia: boolean; command: RecoveryCommand }) =>
      api.trashChat(id, { ...command, delete_generated_media: deleteGeneratedMedia }),
    onMutate: async ({ id: deletedId }) => {
      await client.cancelQueries({ queryKey: ["chats"] });
      const previousChats = snapshotChatPages(client);
      const remainingChats = (chats ?? []).filter((candidate) => candidate.id !== deletedId);
      const previousCurrentChatId = currentChatId;
      const nextChatId = activeChatId === deletedId ? remainingChats.find((candidate) => !candidate.archived)?.id ?? null : currentChatId;
      changeChatPages(client, (item) => item.id === deletedId ? null : item);
      if (activeChatId === deletedId) {
        setCurrentChatId(nextChatId);
        rememberCurrentChat(nextChatId);
      }
      return { previousChats, previousCurrentChatId, nextChatId, wasOpen: activeChatId === deletedId };
    },
    onSuccess: (item, { id: deletedId }, context) => {
      client.removeQueries({ queryKey: ["chat", deletedId] });
      client.removeQueries({ queryKey: ["chat-management", deletedId] });
      setDeleted({ item, wasOpen: context?.wasOpen ?? false, replacementId: context?.nextChatId ?? null });
      undoCommand.current = null;
      undo.reset();
      void client.invalidateQueries({ queryKey: ["recovery-items"] });
      void client.invalidateQueries({ queryKey: ["artifacts"] });
      void client.invalidateQueries({ queryKey: ["artifact-storage"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (_error, _deletedChat, context) => {
      if (!context) return;
      restoreChatPages(client, context.previousChats);
      if (currentChatId === context.nextChatId || currentChatId === context.previousCurrentChatId) {
        setCurrentChatId(context.previousCurrentChatId);
        rememberCurrentChat(context.previousCurrentChatId);
      }
    },
    onSettled: () => void client.invalidateQueries({ queryKey: ["chats"] }),
  });
  return { ...mutation, deleted, undo, dismissUndo: () => {
    if (!undo.isPending) setDeleted(null);
  }, recheckUndo: () => {
    if (!undo.isPending) { undoCommand.current = null; undo.reset(); }
  } };
}
