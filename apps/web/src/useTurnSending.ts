import { useMutation, type QueryClient } from "@tanstack/react-query";
import { useCallback, type Dispatch, type SetStateAction } from "react";
import { api } from "./api";
import type { PendingTurn } from "./chatComposerContracts";
import type { ComposerDraft, ComposerPromptSource } from "./composerPromptSource";
import type { TurnReference } from "./mentionDraft";
import { recoverPromptSourceSend } from "./promptSourceSendRecovery";
import type { ChatDetail, TurnAccepted } from "./types";

export type SendTurnVariables = PendingTurn & {
  chatId: string;
  artifacts: string[];
  settings: Record<string, unknown>;
  /** Subject ids chosen from the mention picker, never parsed from the text. */
  references: TurnReference[];
  outputCount?: number;
  promptSource?: ComposerPromptSource;
  stopCurrent?: boolean;
};

/** Send a turn, and fold an accepted one back into what the screen is reading.

The two belong together because the second is what the first does on success,
and because an accepted turn has to be believed in several places at once: the
open chat gains its two messages, and the lists that summarise chats, jobs,
plans and edited branches all go stale at the same moment. Applying it is
exposed as well as used here, because a turn can also be accepted from a prior
message rather than from the composer. */
export function useTurnSending({
  client,
  requestTurnConfirmation,
  setPendingTurns,
  setComposerDrafts,
}: {
  client: QueryClient;
  requestTurnConfirmation: Parameters<typeof api.sendTurn>[11];
  setPendingTurns: Dispatch<SetStateAction<Record<string, PendingTurn[]>>>;
  // The recovery helper hands this setter a value as well as an updater, so
  // it has to be the whole dispatch rather than the updater half of it.
  setComposerDrafts: Dispatch<SetStateAction<Record<string, ComposerDraft>>>;
}) {
    const applyAcceptedTurn = useCallback((chatId: string, accepted: TurnAccepted, activate = true) => {
      client.setQueryData<ChatDetail>(["chat", chatId], (current) => {
        if (!current) return current;
        const messageIds = new Set(current.messages.map((message) => message.id));
        const acceptedMessages = [accepted.user_message, accepted.assistant_message]
          .filter((message) => !messageIds.has(message.id));
        return {
          ...current,
          active_head_message_id: activate ? accepted.assistant_message.id : current.active_head_message_id,
          messages: [...current.messages, ...acceptedMessages],
        };
      });
      void client.invalidateQueries({ queryKey: ["chat", chatId], exact: true });
      void client.invalidateQueries({ queryKey: ["chats"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
      void client.invalidateQueries({ queryKey: ["work-plans", chatId] });
      void client.invalidateQueries({ queryKey: ["edited-branches", chatId] });
    }, [client]);
    const send = useMutation({
      mutationFn: ({ chatId, id, text, mode, artifacts, settings, references, outputCount, promptSource, stopCurrent }: SendTurnVariables) => {
        if (stopCurrent) return api.stopAndSendTurn(
          chatId, text, mode, artifacts, settings, id, references, outputCount,
          promptSource, requestTurnConfirmation,
        );
        return api.sendTurn(
          chatId, text, mode, artifacts, settings, id, "turns", undefined,
          references, outputCount, promptSource, requestTurnConfirmation,
        );
      },
      onMutate: ({ chatId, id, text, mode }) => {
        setPendingTurns((current) => ({
          ...current,
          [chatId]: [...(current[chatId] ?? []), { id, text, mode }],
        }));
      },
      onSuccess: (accepted, { chatId }) => applyAcceptedTurn(chatId, accepted),
      onError: (_error, variables) => recoverPromptSourceSend(client, setComposerDrafts, variables),
      onSettled: (_accepted, _error, { chatId, id }) => {
        setPendingTurns((current) => {
          const remaining = (current[chatId] ?? []).filter((pending) => pending.id !== id);
          const next = { ...current };
          if (remaining.length) next[chatId] = remaining;
          else delete next[chatId];
          return next;
        });
      },
    });
  return { applyAcceptedTurn, send };
}
