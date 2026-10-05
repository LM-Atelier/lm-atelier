import { useMutation, type QueryClient } from "@tanstack/react-query";
import { useCallback, type Dispatch, type SetStateAction } from "react";
import { api } from "./api";
import { applyAcceptedChatPages } from "./acceptedChatPages";
import type { PendingTurn } from "./chatComposerContracts";
import type { ComposerDraft, ComposerPromptSource } from "./composerPromptSource";
import type { TurnReference } from "./mentionDraft";
import { recoverPromptSourceSend } from "./promptSourceSendRecovery";
import type { SourceFitSelection } from "./sourceFit";
import type { ImageInputRole, TurnAccepted } from "./types";

export type SendTurnVariables = PendingTurn & {
  chatId: string;
  artifacts: string[];
  settings: Record<string, unknown>;
  /** Subject ids chosen from the mention picker, never parsed from the text. */
  references: TurnReference[];
  outputCount?: number;
  promptSource?: ComposerPromptSource;
  sourceFit?: SourceFitSelection;
  imageRoles?: ImageInputRole[];
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
      applyAcceptedChatPages(client, chatId, accepted, activate);
      void client.invalidateQueries({ queryKey: ["chat", chatId] });
      void client.invalidateQueries({ queryKey: ["chats"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
      void client.invalidateQueries({ queryKey: ["work-plans", chatId] });
      void client.invalidateQueries({ queryKey: ["edited-branches", chatId] });
    }, [client]);
    const send = useMutation({
      mutationFn: ({ chatId, id, text, mode, artifacts, settings, references, outputCount, promptSource, sourceFit, imageRoles, stopCurrent }: SendTurnVariables) => {
        if (imageRoles !== undefined) {
          if (stopCurrent) return api.stopAndSendTurn(chatId, text, mode, artifacts, settings, id,
            references, outputCount, promptSource, requestTurnConfirmation, sourceFit, imageRoles);
          return api.sendTurn(chatId, text, mode, artifacts, settings, id, "turns", undefined,
            references, outputCount, promptSource, requestTurnConfirmation, sourceFit, undefined, imageRoles);
        }
        // A fitted canvas rides as the last argument only when there is one, so
        // an ordinary send keeps exactly the call it always made.
        const fit = sourceFit ? [sourceFit] as const : [] as const;
        if (stopCurrent) return api.stopAndSendTurn(
          chatId, text, mode, artifacts, settings, id, references, outputCount,
          promptSource, requestTurnConfirmation, ...fit,
        );
        return api.sendTurn(
          chatId, text, mode, artifacts, settings, id, "turns", undefined,
          references, outputCount, promptSource, requestTurnConfirmation, ...fit,
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
