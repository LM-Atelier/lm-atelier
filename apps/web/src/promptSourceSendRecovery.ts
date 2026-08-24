import type { Dispatch, SetStateAction } from "react";
import type { QueryClient } from "@tanstack/react-query";

import {
  EMPTY_COMPOSER_DRAFT,
  type ComposerDraft,
  type ComposerPromptSource,
} from "./composerPromptSource";

export function recoverPromptSourceSend(
  client: QueryClient,
  setComposerDrafts: Dispatch<SetStateAction<Record<string, ComposerDraft>>>,
  variables: { chatId: string; text: string; promptSource?: ComposerPromptSource },
): void {
  if (!variables.promptSource) return;
  const promptSource = variables.promptSource;
  setComposerDrafts((current) => {
    const existing = current[variables.chatId] ?? EMPTY_COMPOSER_DRAFT;
    if (existing.text.trim() || existing.promptSource) return current;
    return {
      ...current,
      [variables.chatId]: { text: variables.text, promptSource },
    };
  });
  void client.invalidateQueries({ queryKey: ["prompt-batch", promptSource.batch_id] });
}
