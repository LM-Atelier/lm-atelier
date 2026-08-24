import { useCallback, type Dispatch, type SetStateAction } from "react";
import type { QueryClient } from "@tanstack/react-query";
import { insertedComposerDraft, type ComposerDraft, type PromptComposerInsertion } from "./composerPromptSource";
import type { View } from "./rooms";
import type { Chat, ChatDetail } from "./types";
import { focusMainContent } from "./viewHelpers";

interface PromptLibraryComposerInsertionOptions {
  client: QueryClient;
  setComposerDrafts: Dispatch<SetStateAction<Record<string, ComposerDraft>>>;
  setChatDrafts: Dispatch<SetStateAction<Record<string, Partial<Chat>>>>;
  persistImageMode: (chatId: string) => void;
  setCurrentChatId: Dispatch<SetStateAction<string | null>>;
  setView: Dispatch<SetStateAction<View>>;
  currentChatStorageKey: string;
}

export function usePromptLibraryComposerInsertion({
  client,
  setComposerDrafts,
  setChatDrafts,
  persistImageMode,
  setCurrentChatId,
  setView,
  currentChatStorageKey,
}: PromptLibraryComposerInsertionOptions) {
  return useCallback((chatId: string, insertion: PromptComposerInsertion) => {
    setComposerDrafts((current) => ({
      ...current,
      [chatId]: insertedComposerDraft(insertion),
    }));
    setChatDrafts((current) => ({
      ...current,
      [chatId]: { ...(current[chatId] ?? {}), routing_mode: "image" },
    }));
    client.setQueryData<ChatDetail>(["chat", chatId], (current) => (
      current ? { ...current, routing_mode: "image" } : current
    ));
    persistImageMode(chatId);
    setCurrentChatId(chatId);
    localStorage.setItem(currentChatStorageKey, chatId);
    setView("chat");
    focusMainContent();
  }, [
    client,
    currentChatStorageKey,
    persistImageMode,
    setChatDrafts,
    setComposerDrafts,
    setCurrentChatId,
    setView,
  ]);
}
